r""" Helper functions """
import os
import random
import numpy as np
from PIL import Image, ImageDraw

import torch
import numpy as np
import torch.distributed as dist

def fix_randseed(seed):
    r""" Set random seeds for reproducibility """
    if seed is None:
        seed = int(random.random() * 1e5)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def mean(x):
    return sum(x) / len(x) if len(x) > 0 else 0.0


def to_cuda(batch):
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.cuda()
    return batch


def to_cpu(tensor):
    return tensor.detach().clone().cpu()

def find_free_port():
    """
    Used for distributed learning
    """
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def reduce_metric(metric, average=True):
    world_size = get_world_size()
    if world_size < 2:
        return metric
    with torch.no_grad():
        if not metric.is_contiguous():
            metric = metric.contiguous()
        # dist.barrier()
        dist.all_reduce(metric)
        if average:
            metric /= world_size
    return metric

def get_stroke_preset(stroke_preset):
        if stroke_preset == 'rand_curve':
            return {
                "nVertexBound": [10, 30],
                "maxHeadSpeed": 20,
                "maxHeadAcceleration": (15, 0.5),
                "brushWidthBound": (3, 10),
                "nMovePointRatio": 0.5,
                "maxPiontMove": 3,
                "maxLineAcceleration": (5, 0.5),
                "boarderGap": None,
                "maxInitSpeed": 6
            }
        elif stroke_preset == 'rand_curve_small':
            return {
                "nVertexBound": [6, 22],
                "maxHeadSpeed": 12,
                "maxHeadAcceleration": (8, 0.5),
                "brushWidthBound": (2.5, 5),
                "nMovePointRatio": 0.5,
                "maxPiontMove": 1.5,
                "maxLineAcceleration": (3, 0.5),
                "boarderGap": None,
                "maxInitSpeed": 3
            }
        else:
            raise NotImplementedError(f'The stroke presetting "{stroke_preset}" does not exist.')

def get_random_points_from_mask(mask, n=5):
        h,w = mask.shape
        view_mask = mask.reshape(h*w)
        non_zero_idx = view_mask.nonzero()[:,0]
        selected_idx = torch.randperm(len(non_zero_idx))[:n]
        non_zero_idx = non_zero_idx[selected_idx]
        # import pdb; pdb.set_trace()
        y = torch.div(non_zero_idx , w)
        x = (non_zero_idx % w)
        return torch.cat((x[:,None], y[:,None]), dim=1).cpu().numpy()


def get_mask_by_input_strokes(
    init_points, imageWidth=320, imageHeight=180, nStroke=5,
    nVertexBound=[10, 30], maxHeadSpeed=15, maxHeadAcceleration=(15, 0.5),
    brushWidthBound=(5, 20), boarderGap=None, nMovePointRatio=0.5, maxPiontMove=10,
    maxLineAcceleration=5, maxInitSpeed=5
):
    '''
    Get video masks by random strokes which move randomly between each
    frame, including the whole stroke and its control points

    Parameters
    ----------
        imageWidth: Image width
        imageHeight: Image height
        nStroke: Number of drawed lines
        nVertexBound: Lower/upper bound of number of control points for each line
        maxHeadSpeed: Max head speed when creating control points
        maxHeadAcceleration: Max acceleration applying on the current head point (
            a head point and its velosity decides the next point)
        brushWidthBound (min, max): Bound of width for each stroke
        boarderGap: The minimum gap between image boarder and drawed lines
        nMovePointRatio: The ratio of control points to move for next frames
        maxPiontMove: The magnitude of movement for control points for next frames
        maxLineAcceleration: The magnitude of acceleration for the whole line

    Examples
    ----------
        object_like_setting = {
            "nVertexBound": [5, 20],
            "maxHeadSpeed": 15,
            "maxHeadAcceleration": (15, 3.14),
            "brushWidthBound": (30, 50),
            "nMovePointRatio": 0.5,
            "maxPiontMove": 10,
            "maxLineAcceleration": (5, 0.5),
            "boarderGap": 20,
            "maxInitSpeed": 10,
        }
        rand_curve_setting = {
            "nVertexBound": [10, 30],
            "maxHeadSpeed": 20,
            "maxHeadAcceleration": (15, 0.5),
            "brushWidthBound": (3, 10),
            "nMovePointRatio": 0.5,
            "maxPiontMove": 3,
            "maxLineAcceleration": (5, 0.5),
            "boarderGap": 20,
            "maxInitSpeed": 6
        }
        get_video_masks_by_moving_random_stroke(video_len=5, nStroke=3, **object_like_setting)
    '''
    # Initilize a set of control points to draw the first mask
    # import pdb; pdb.set_trace()
    mask = Image.new(mode='1', size=(imageWidth, imageHeight), color=1)
    control_points_set = []
    for i in range(nStroke):
        brushWidth = np.random.randint(brushWidthBound[0], brushWidthBound[1])
        Xs, Ys, velocity = get_random_stroke_control_points(
            init_point=init_points[i],
            imageWidth=imageWidth, imageHeight=imageHeight,
            nVertexBound=nVertexBound, maxHeadSpeed=maxHeadSpeed,
            maxHeadAcceleration=maxHeadAcceleration, boarderGap=boarderGap,
            maxInitSpeed=maxInitSpeed
        )
        control_points_set.append((Xs, Ys, velocity, brushWidth))
        draw_mask_by_control_points(mask, Xs, Ys, brushWidth, fill=0)

    # Generate the following masks by randomly move strokes and their control points
    mask = Image.new(mode='1', size=(imageWidth, imageHeight), color=1)
    for j in range(len(control_points_set)):
        Xs, Ys, velocity, brushWidth = control_points_set[j]
        new_Xs, new_Ys = random_move_control_points(
            Xs, Ys, velocity, nMovePointRatio, maxPiontMove,
            maxLineAcceleration, boarderGap
        )
        control_points_set[j] = (new_Xs, new_Ys, velocity, brushWidth)
    for Xs, Ys, velocity, brushWidth in control_points_set:
        draw_mask_by_control_points(mask, Xs, Ys, brushWidth, fill=0)

    return np.array(mask)


def random_accelerate(velocity, maxAcceleration, dist='uniform'):
    speed, angle = velocity
    d_speed, d_angle = maxAcceleration

    if dist == 'uniform':
        speed += np.random.uniform(-d_speed, d_speed)
        angle += np.random.uniform(-d_angle, d_angle)
    elif dist == 'guassian':
        speed += np.random.normal(0, d_speed / 2)
        angle += np.random.normal(0, d_angle / 2)
    else:
        raise NotImplementedError(f'Distribution type {dist} is not supported.')

    return (speed, angle)


def random_move_control_points(Xs, Ys, lineVelocity, nMovePointRatio, maxPiontMove, maxLineAcceleration, boarderGap=15):
    new_Xs = Xs.copy()
    new_Ys = Ys.copy()

    # move the whole line and accelerate
    speed, angle = lineVelocity
    new_Xs += int(speed * np.cos(angle))
    new_Ys += int(speed * np.sin(angle))
    lineVelocity = random_accelerate(lineVelocity, maxLineAcceleration, dist='guassian')

    # choose points to move
    chosen = np.arange(len(Xs))
    np.random.shuffle(chosen)
    chosen = chosen[:int(len(Xs) * nMovePointRatio)]
    for i in chosen:
        new_Xs[i] += np.random.randint(-maxPiontMove, maxPiontMove)
        new_Ys[i] += np.random.randint(-maxPiontMove, maxPiontMove)
    return new_Xs, new_Ys


def get_random_stroke_control_points(
    init_point,
    imageWidth, imageHeight,
    nVertexBound=(10, 30), maxHeadSpeed=10, maxHeadAcceleration=(5, 0.5), boarderGap=20,
    maxInitSpeed=10
):
    '''
    Implementation the free-form training masks generating algorithm
    proposed by JIAHUI YU et al. in "Free-Form Image Inpainting with Gated Convolution"
    '''
    startX = init_point[0]
    startY = init_point[1]

    Xs = [init_point[0]]
    Ys = [init_point[1]]

    numVertex = np.random.randint(nVertexBound[0], nVertexBound[1])

    angle = np.random.uniform(0, 2 * np.pi)
    speed = np.random.uniform(0, maxHeadSpeed)

    for i in range(numVertex):
        speed, angle = random_accelerate((speed, angle), maxHeadAcceleration)
        speed = np.clip(speed, 0, maxHeadSpeed)

        nextX = startX + speed * np.sin(angle)
        nextY = startY + speed * np.cos(angle)

        if boarderGap is not None:
            nextX = np.clip(nextX, boarderGap, imageWidth - boarderGap)
            nextY = np.clip(nextY, boarderGap, imageHeight - boarderGap)

        startX, startY = nextX, nextY
        Xs.append(nextX)
        Ys.append(nextY)

    velocity = get_random_velocity(maxInitSpeed, dist='guassian')

    return np.array(Xs), np.array(Ys), velocity


def get_random_velocity(max_speed, dist='uniform'):
    if dist == 'uniform':
        speed = np.random.uniform(max_speed)
    elif dist == 'guassian':
        speed = np.abs(np.random.normal(0, max_speed / 2))
    else:
        raise NotImplementedError(f'Distribution type {dist} is not supported.')

    angle = np.random.uniform(0, 2 * np.pi)
    return (speed, angle)


def draw_mask_by_control_points(mask, Xs, Ys, brushWidth, fill=255):
    radius = brushWidth // 2 - 1
    for i in range(1, len(Xs)):
        draw = ImageDraw.Draw(mask)
        startX, startY = Xs[i - 1], Ys[i - 1]
        nextX, nextY = Xs[i], Ys[i]
        draw.line((startX, startY) + (nextX, nextY), fill=fill, width=brushWidth)
    for x, y in zip(Xs, Ys):
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fill)
    return mask


# modified from https://github.com/naoto0804/pytorch-inpainting-with-partial-conv/blob/master/generate_data.py
def get_random_walk_mask(imageWidth=320, imageHeight=180, length=None):
    action_list = [[0, 1], [0, -1], [1, 0], [-1, 0]]
    canvas = np.zeros((imageHeight, imageWidth)).astype("i")
    if length is None:
        length = imageWidth * imageHeight
    x = random.randint(0, imageHeight - 1)
    y = random.randint(0, imageWidth - 1)
    x_list = []
    y_list = []
    for i in range(length):
        r = random.randint(0, len(action_list) - 1)
        x = np.clip(x + action_list[r][0], a_min=0, a_max=imageHeight - 1)
        y = np.clip(y + action_list[r][1], a_min=0, a_max=imageWidth - 1)
        x_list.append(x)
        y_list.append(y)
    canvas[np.array(x_list), np.array(y_list)] = 1
    return Image.fromarray(canvas * 255).convert('1')


def get_masked_ratio(mask):
    """
    Calculate the masked ratio.
    mask: Expected a binary PIL image, where 0 and 1 represent
          masked(invalid) and valid pixel values.
    """
    hist = mask.histogram()
    return hist[0] / np.prod(mask.size)