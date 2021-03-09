import argparse
import math
import os
import random

import pydiffvg
import torch
import skimage
import skimage.io

import clip_utils


pydiffvg.set_print_timing(True)

gamma = 1.0

def main(args):
    outdir = os.path.join(args.results_dir, args.prompt, args.subdir)

    # Use GPU if available
    pydiffvg.set_use_gpu(torch.cuda.is_available())
    
    canvas_width, canvas_height = 224, 224
    num_paths = args.num_paths
    max_width = args.max_width
    
    random.seed(1234)
    torch.manual_seed(1234)
    
    shapes = []
    shape_groups = []
    if args.use_blob:
        for i in range(num_paths):
            num_segments = random.randint(3, 5)
            num_control_points = torch.zeros(num_segments, dtype = torch.int32) + 2
            points = []
            p0 = (random.random(), random.random())
            points.append(p0)
            for j in range(num_segments):
                radius = 0.05
                p1 = (p0[0] + radius * (random.random() - 0.5), p0[1] + radius * (random.random() - 0.5))
                p2 = (p1[0] + radius * (random.random() - 0.5), p1[1] + radius * (random.random() - 0.5))
                p3 = (p2[0] + radius * (random.random() - 0.5), p2[1] + radius * (random.random() - 0.5))
                points.append(p1)
                points.append(p2)
                if j < num_segments - 1:
                    points.append(p3)
                    p0 = p3
            points = torch.tensor(points)
            points[:, 0] *= canvas_width
            points[:, 1] *= canvas_height
            path = pydiffvg.Path(num_control_points = num_control_points,
                                 points = points,
                                 stroke_width = torch.tensor(1.0),
                                 is_closed = True)
            shapes.append(path)
            path_group = pydiffvg.ShapeGroup(shape_ids = torch.tensor([len(shapes) - 1]),
                                             fill_color = torch.tensor([random.random(),
                                                                        random.random(),
                                                                        random.random(),
                                                                        random.random()]))
            shape_groups.append(path_group)
    else:
        for i in range(num_paths):
            num_segments = random.randint(1, 3)
            num_control_points = torch.zeros(num_segments, dtype = torch.int32) + 2
            points = []
            p0 = (random.random(), random.random())
            points.append(p0)
            for j in range(num_segments):
                radius = 0.05
                p1 = (p0[0] + radius * (random.random() - 0.5), p0[1] + radius * (random.random() - 0.5))
                p2 = (p1[0] + radius * (random.random() - 0.5), p1[1] + radius * (random.random() - 0.5))
                p3 = (p2[0] + radius * (random.random() - 0.5), p2[1] + radius * (random.random() - 0.5))
                points.append(p1)
                points.append(p2)
                points.append(p3)
                p0 = p3
            points = torch.tensor(points)
            points[:, 0] *= canvas_width
            points[:, 1] *= canvas_height
            #points = torch.rand(3 * num_segments + 1, 2) * min(canvas_width, canvas_height)
            path = pydiffvg.Path(num_control_points = num_control_points,
                                 points = points,
                                 stroke_width = torch.tensor(1.0),
                                 is_closed = False)
            shapes.append(path)
            path_group = pydiffvg.ShapeGroup(shape_ids = torch.tensor([len(shapes) - 1]),
                                             fill_color = None,
                                             stroke_color = torch.tensor([random.random(),
                                                                          random.random(),
                                                                          random.random(),
                                                                          random.random()]))
            shape_groups.append(path_group)
    
    scene_args = pydiffvg.RenderFunction.serialize_scene(\
        canvas_width, canvas_height, shapes, shape_groups)
    
    render = pydiffvg.RenderFunction.apply
    img = render(canvas_width, # width
                 canvas_height, # height
                 2,   # num_samples_x
                 2,   # num_samples_y
                 0,   # seed
                 None,
                 *scene_args)
    pydiffvg.imwrite(img.cpu(), os.path.join(outdir, 'init.png'), gamma=gamma)

    points_vars = []
    stroke_width_vars = []
    color_vars = []
    for path in shapes:
        path.points.requires_grad = True
        points_vars.append(path.points)
    if not args.use_blob:
        for path in shapes:
            path.stroke_width.requires_grad = True
            stroke_width_vars.append(path.stroke_width)
    if args.use_blob:
        for group in shape_groups:
            group.fill_color.requires_grad = True
            color_vars.append(group.fill_color)
    else:
        for group in shape_groups:
            group.stroke_color.requires_grad = True
            color_vars.append(group.stroke_color)

    # Embed prompt
    text_features = clip_utils.embed_text(args.prompt)

    # Optimize
    losses = []
    points_optim = torch.optim.Adam(points_vars, lr=args.points_lr)
    if len(stroke_width_vars) > 0:
        width_optim = torch.optim.Adam(stroke_width_vars, lr=args.width_lr)
    color_optim = torch.optim.Adam(color_vars, lr=args.color_lr)
    # Adam iterations.
    for t in range(args.num_iter):
        print('iteration:', t)
        points_optim.zero_grad()
        if len(stroke_width_vars) > 0:
            width_optim.zero_grad()
        color_optim.zero_grad()
        # Forward pass: render the image.
        scene_args = pydiffvg.RenderFunction.serialize_scene(\
            canvas_width, canvas_height, shapes, shape_groups)
        img = render(canvas_width, # width
                     canvas_height, # height
                     2,   # num_samples_x
                     2,   # num_samples_y
                     t,   # seed
                     None,
                     *scene_args)
        # Save the intermediate render.
        pydiffvg.imwrite(img.cpu(), os.path.join(outdir, 'iter_{}.png'.format(t)), gamma=gamma)
        image_features = clip_utils.embed_image(img)
        loss = -torch.cosine_similarity(text_features, image_features, dim=-1).mean()
        print('render loss:', loss.item())
    
        # Backpropagate the gradients.
        loss.backward()
        losses.append(loss.item())

        # Take a gradient descent step.
        points_optim.step()
        if len(stroke_width_vars) > 0:
            width_optim.step()
        color_optim.step()
        if len(stroke_width_vars) > 0:
            for path in shapes:
                path.stroke_width.data.clamp_(1.0, max_width)
        if args.use_blob:
            for group in shape_groups:
                group.fill_color.data.clamp_(0.0, 1.0)
        else:
            for group in shape_groups:
                group.stroke_color.data.clamp_(0.0, 1.0)

        if t % 10 == 0 or t == args.num_iter - 1:
            pydiffvg.save_svg(os.path.join(outdir, 'iter_{}.svg'.format(t)),
                              canvas_width, canvas_height, shapes, shape_groups)
            clip_utils.plot_losses(losses, outdir)
    
    # Render the final result.
    img = render(args.final_px, # width
                 args.final_px, # height
                 2,   # num_samples_x
                 2,   # num_samples_y
                 0,   # seed
                 None,
                 *scene_args)
    # Save the intermediate render.
    pydiffvg.imwrite(img.cpu(), os.path.join(outdir, 'final.png'), gamma=gamma)
    # Convert the intermediate renderings to a video with a white background.
    from subprocess import call
    call(["ffmpeg", "-framerate", "24", "-i",
        os.path.join(outdir, "iter_%d.png"), "-vb", "20M", "-filter_complex",
        "color=white,format=rgb24[c];[c][0]scale2ref[c][i];[c][i]overlay=format=auto:shortest=1,setsar=1",
        os.path.join(outdir, "out.mp4")])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", help="text to use for image generation loss")
    parser.add_argument('--results_dir', default='results/text_to_painting')
    parser.add_argument('--subdir', default='default')
    parser.add_argument("--num_paths", type=int, default=512)
    parser.add_argument("--max_width", type=float, default=2.0)
    parser.add_argument("--final_px", type=int, default=512)
    parser.add_argument("--num_iter", type=int, default=500)
    parser.add_argument("--use_blob", dest='use_blob', action='store_true')
    parser.add_argument("--points_lr", type=float, default=1.0)
    parser.add_argument("--width_lr", type=float, default=0.1)
    parser.add_argument("--color_lr", type=float, default=0.01)
    args = parser.parse_args()
    main(args)
