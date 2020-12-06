import torch
import torch.nn as nn
import torch.nn.functional as F


from model.matcher import Matcher
from model.box_regression import Box2BoxTransform
from utils.box_utils import pairwise_iou, cat_boxes


class RetinaLoss(nn.Module):
    def __init__(
        self,
        num_classes=80,
        focal_loss_alpha=0.25,
        focal_loss_gamma=2.0,
        smooth_l1_beta=0.1,
        test_score_thresh=0.05,
        test_topk_candidates=1000,
        test_nms_thresh=0.5,
        max_detection_per_image=100,
    ):
        super().__init__()
        self.anchor_matcher = Matcher([0.4, 0.5], [-1, 0, 1])
        self.box2box_transform = Box2BoxTransform([1.0, 1.0, 1.0, 1.0])
        self.num_classes = num_classes

        # Loss params.
        self.focal_loss_alpha = focal_loss_alpha
        self.focal_loss_gamma = focal_loss_gamma
        self.smooth_l1_beta = smooth_l1_beta

        # Inference params.
        self.test_score_thresh = test_score_thresh
        self.test_topk_candidates = test_topk_candidates
        self.test_nms_thresh = test_nms_thresh
        self.max_detection_per_image = max_detection_per_image

        """
        In Detectron1, loss is normalized by number of foreground samples in the batch.
        When batch size is 1 per GPU, #foreground has a large variance and
        using it lead to lower performance. Here we maintain an EMA of #foreground to
        stabilize the normalizer.
        """
        self.loss_normalizer = (
            100  # initialize with any reasonable #fg that's not too small
        )
        self.loss_normalizer_momentum = 0.9

    def forward(self, pred_logits, pred_anchor_deltas, anchors, boxes, labels):
        gt_labels, gt_boxes = self.label_anchors(anchors, boxes, labels)
        losses = self.losses(
            anchors, pred_logits, gt_labels, pred_anchor_deltas, gt_boxes
        )
        return losses

    def label_anchors(self, anchors, boxes, labels):
        anchors = cat_boxes(anchors)
        gt_labels = []
        matched_gt_boxes = []
        bs = len(boxes)

        for i in range(bs):
            ## pdb.set_trace()
            matched_quality_matrix = pairwise_iou(boxes[i], anchors)
            matched_idxs, anchor_labels = self.anchor_matcher(matched_quality_matrix)
            ## pdb.set_trace()
            del matched_quality_matrix

            if len(boxes[i]) > 0:
                matched_gt_boxes_i = boxes[i][matched_idxs]
                gt_labels_i = labels[i][matched_idxs]
                # Label 0 means background.
                gt_labels_i[anchor_labels == 0] = self.num_classes
                # Label -1 means ignore.
                gt_labels_i[anchor_labels == -1] = -1

            else:
                matched_gt_boxes_i = torch.zeros_like(anchors)
                gt_labels_i = torch.zeros_like(matched_idxs) + self.num_classes

            gt_labels.append(gt_labels_i)
            matched_gt_boxes.append(matched_gt_boxes_i)

        return gt_labels, matched_gt_boxes

    def losses(self, anchors, pred_logits, gt_labels, pred_anchor_deltas, gt_boxes):
        """
        Args:
            anchors (list[Tensors]): a list of feature maps for each level.
            gt_labels, gt_boxes: see output of :meth: `self.label_anchors`.
                Their shapes are (N, R) and (N, R, 4), respectively, where R
                is the total number of anchors across all feature levels.
                i.e. sum(Hi x Wi x Ai)
            pred_logits, pred_anchor_deltas: both are list[Tensor]. Each
                element in the list corresponds to one level and has shape
                (N, Hi x Wi x Ai, K or 4) - where K is the number of classes.

        Returns:
            dict[str, Tensor]:
                mapping from named loss to a scalar tensor storing the
                loss for classification and bbox regression. Used during
                training only. The dict keys are: "loss_cls" and "loss_box_reg".
        """

        num_images = len(gt_labels)
        gt_labels = torch.stack(gt_labels).squeeze(-1)  # (N, R)
        anchors = cat_boxes(anchors)
        gt_anchor_deltas = [
            self.box2box_transform.get_deltas(anchors, k) for k in gt_boxes
        ]
        gt_anchor_deltas = torch.stack(gt_anchor_deltas)  # (N, R, 4)

        valid_mask = gt_labels >= 0
        pos_mask = (gt_labels >= 0) & (gt_labels != self.num_classes)
        num_pos_anchors = pos_mask.sum().item()
        self.loss_normalizer = self.loss_normalizer_momentum * self.loss_normalizer + (
            1 - self.loss_normalizer_momentum
        ) * max(num_pos_anchors, 1)

        # Classification and regression loss.
        gt_labels_target = F.one_hot(
            gt_labels[valid_mask], num_classes=self.num_classes + 1
        )[
            :, :-1
        ]  # no loss for the last (bg) class
        loss_cls = sigmoid_focal_loss(
            torch.cat(pred_logits, dim=1)[valid_mask],
            gt_labels_target.to(pred_logits[0].dtype),
            alpha=self.focal_loss_alpha,
            gamma=self.focal_loss_gamma,
            reduction="sum",
        )

        loss_box_reg = smooth_l1_loss(
            torch.cat(pred_anchor_deltas, dim=1)[pos_mask],
            gt_anchor_deltas[pos_mask],
            beta=self.smooth_l1_beta,
            reduction="sum",
        )
        return {
            "loss_cls": loss_cls / self.loss_normalizer,
            "loss_box_reg": loss_box_reg / self.loss_normalizer,
        }


def sigmoid_focal_loss(
    inputs: torch.Tensor, targets: torch.Tensor, gamma=2, alpha=0.25, reduction="none"
) -> torch.Tensor:
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)

    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()

    elif reduction == "sum":
        loss = loss.sum()

    return loss


def smooth_l1_loss(
    input: torch.Tensor, target: torch.Tensor, beta: float, reduction: str = "none"
) -> torch.Tensor:
    """
    Smooth L1 loss defined in the Fast R-CNN paper as:
                  | 0.5 * x ** 2 / beta   if abs(x) < beta
    smoothl1(x) = |
                  | abs(x) - 0.5 * beta   otherwise,
    where x = input - target.
    Smooth L1 loss is related to Huber loss, which is defined as:
                | 0.5 * x ** 2                  if abs(x) < beta
     huber(x) = |
                | beta * (abs(x) - 0.5 * beta)  otherwise
    Smooth L1 loss is equal to huber(x) / beta. This leads to the following
    differences:
     - As beta -> 0, Smooth L1 loss converges to L1 loss, while Huber loss
       converges to a constant 0 loss.
     - As beta -> +inf, Smooth L1 converges to a constant 0 loss, while Huber loss
       converges to L2 loss.
     - For Smooth L1 loss, as beta varies, the L1 segment of the loss has a constant
       slope of 1. For Huber loss, the slope of the L1 segment is beta.
    Smooth L1 loss can be seen as exactly L1 loss, but with the abs(x) < beta
    portion replaced with a quadratic function such that at abs(x) = beta, its
    slope is 1. The quadratic segment smooths the L1 loss near x = 0.
    Args:
        input (Tensor): input tensor of any shape
        target (Tensor): target value tensor with the same shape as input
        beta (float): L1 to L2 change point.
            For beta values < 1e-5, L1 loss is computed.
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
    Returns:
        The loss with the reduction option applied.
    Note:
        PyTorch's builtin "Smooth L1 loss" implementation does not actually
        implement Smooth L1 loss, nor does it implement Huber loss. It implements
        the special case of both in which they are equal (beta=1).
        See: https://pytorch.org/docs/stable/nn.html#torch.nn.SmoothL1Loss.
    """
    if beta < 1e-5:
        # if beta == 0, then torch.where will result in nan gradients when
        # the chain rule is applied due to pytorch implementation details
        # (the False branch "0.5 * n ** 2 / 0" has an incoming gradient of
        # zeros, rather than "no gradient"). To avoid this issue, we define
        # small values of beta to be exactly l1 loss.
        loss = torch.abs(input - target)
    else:
        n = torch.abs(input - target)
        cond = n < beta
        loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    return loss