import torch
import torch.nn as nn
import torch.nn.functional as F

def bce_loss(input, target, reduce=True):
    """
    Numerically stable version of the binary cross-entropy loss function.
    As per https://github.com/pytorch/pytorch/issues/751
    See the TensorFlow docs for a derivation of this formula:
    https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits
    Inputs:
    - input: PyTorch Tensor of shape (N, ) giving scores.
    - target: PyTorch Tensor of shape (N,) containing 0 and 1 giving targets.
    Returns:
    - A PyTorch Tensor containing the mean BCE loss over the minibatch of
      input data.
    """
    neg_abs = -input.abs()
    loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
    if reduce:
        return loss.mean()
    else:
        return loss


def calculate_iou_3d(pred_boxes, gt_boxes):
    # Extracting centroids and dimensions for predicted boxes
    pred_cx, pred_cy, pred_cz, pred_w, pred_l, pred_h, pred_alpha = pred_boxes[:, 0], pred_boxes[:, 1], pred_boxes[:, 2], pred_boxes[:, 3], pred_boxes[:, 4], pred_boxes[:, 5], pred_boxes[:, 6]
    # Extracting centroids and dimensions for ground truth boxes
    gt_cx, gt_cy, gt_cz, gt_w, gt_l, gt_h, gt_alpha = gt_boxes[:, 0], gt_boxes[:, 1], gt_boxes[:, 2], gt_boxes[:, 3], gt_boxes[:, 4], gt_boxes[:, 5], gt_boxes[:, 6]

    # Calculate cosine and sine of alpha
    cos_alpha = torch.cos(pred_alpha)
    sin_alpha = torch.sin(pred_alpha)

    # Rotate coordinates of ground truth boxes to align with predicted boxes
    x1, y1 = pred_cx.unsqueeze(1), pred_cy.unsqueeze(1)
    x2, y2 = gt_cx.unsqueeze(0), gt_cy.unsqueeze(0)
    x2_rot = (x2 - x1) * cos_alpha.unsqueeze(0) - (y2 - y1) * sin_alpha.unsqueeze(0) + x1
    y2_rot = (x2 - x1) * sin_alpha.unsqueeze(0) + (y2 - y1) * cos_alpha.unsqueeze(0) + y1

    # Calculate intersection
    dx = torch.abs(x2_rot - x2)
    dy = torch.abs(y2_rot - y2)
    dz = torch.abs(pred_cz.unsqueeze(1) - gt_cz.unsqueeze(0))
    intersection = torch.zeros_like(dx)
    intersection += torch.where(dx <= (pred_w.unsqueeze(1) + gt_w.unsqueeze(0)) / 2, 1, 0)
    intersection *= torch.where(dy <= (pred_l.unsqueeze(1) + gt_l.unsqueeze(0)) / 2, 1, 0)
    intersection *= torch.where(dz <= (pred_h.unsqueeze(1) + gt_h.unsqueeze(0)) / 2, 1, 0)

    # Calculate volumes
    vol1 = pred_w * pred_l * pred_h
    vol2 = gt_w * gt_l * gt_h

    # Calculate union
    union = vol1.unsqueeze(1) + vol2.unsqueeze(0) - intersection

    # Calculate IOU
    iou = intersection / union
    return iou

def iou_loss_3d(pred_boxes, gt_boxes):
    iou = calculate_iou_3d(pred_boxes, gt_boxes)
    iou_loss = 1 - iou
    return iou_loss.mean()


def calculate_model_losses(args, target, pred, name, angles=None, angles_pred=None, mu=None, logvar=None,
                           KL_weight=None, writer=None, counter=None, withangles=False):
    total_loss = 0.0
    losses = {}
    rec_loss = F.l1_loss(pred, target)
    
    total_loss = add_loss(total_loss, rec_loss, losses, name, 1)
    if withangles:
        angle_loss = F.nll_loss(angles_pred, angles)
        total_loss = add_loss(total_loss, angle_loss, losses, 'angle_pred', 1)

        angles_gt = angles.unsqueeze(1)
        iou_gt = torch.concat((target,angles_gt),1)
        angles_pred_ = -180 + (torch.argmax(angles_pred, dim=1, keepdim=True) + 1)* 15.0
        iou_pred = torch.concat((pred,angles_pred_),1)
        iou_loss = iou_loss_3d(iou_pred, iou_gt)
        total_loss = add_loss(total_loss, iou_loss, losses, 'iou', 0.01)
    try:
        loss_gauss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.size(0)

    except:
        print("blowup!!!")
        print("logvar", torch.sum(logvar.data), torch.sum(torch.abs(logvar.data)), torch.max(logvar.data),
              torch.min(logvar.data))
        print("mu", torch.sum(mu.data), torch.sum(torch.abs(mu.data)), torch.max(mu.data), torch.min(mu.data))
        return total_loss, losses
    total_loss = add_loss(total_loss, loss_gauss, losses, 'KLD_Gauss', KL_weight)

    writer.add_scalar('Train_Loss_KL_{}'.format(name), loss_gauss, counter)
    writer.add_scalar('Train_Loss_Rec_{}'.format(name), rec_loss, counter)
    if withangles:
        writer.add_scalar('Train_Loss_Angle_{}'.format(name), angle_loss, counter)
    return total_loss, losses


def add_loss(total_loss, curr_loss, loss_dict, loss_name, weight=1):
    curr_loss_weighted = curr_loss * weight
    loss_dict[loss_name] = curr_loss_weighted.item()
    if total_loss is not None:
        return total_loss + curr_loss_weighted
    else:
        return curr_loss_weighted
    retur

class VQLoss(nn.Module):
    def __init__(self, codebook_weight=1.0):
        super().__init__()
        self.codebook_weight = codebook_weight

    def forward(self, codebook_loss, inputs, reconstructions, split="train"):
        rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous())

        nll_loss = rec_loss
        nll_loss = torch.mean(nll_loss)

        loss = nll_loss + self.codebook_weight * codebook_loss.mean()

        log = {
            "loss_total": loss.clone().detach().mean(),
            "loss_codebook": codebook_loss.detach().mean(),
            "loss_nll": nll_loss.detach().mean(),
            "loss_rec": rec_loss.detach().mean(),
        }

        return loss, log