# Baseline failure analysis

Baseline: YOLO-World `yolov8s-worldv2`, prompt "cardboard box", confidence
0.25, IoU match 0.5. Test split: 57 images, 68 true boxes.
Result: 45 TP, 68 FP, 23 FN. Precision 0.40, recall 0.66, F1 0.50.

Every false positive (68) and every miss (23) was cropped and inspected
by hand. The category counts below come from that full pass, not a sample.
In the images here, green = ground truth, red = prediction with confidence.

## False positives: 68 phantom detections

| category | count | share |
|---|---|---|
| Duplicate or fragment detection on a box already detected (a flap, one face, or the same box twice) | 21 | 31% |
| Background texture and soft objects: plastic-wrapped mattress, bags, blankets, paper, a jar | 17 | 25% |
| Box-shaped non-cardboard container: plastic tote (7), red toolbox or wagon (5), foam cooler (4) | 16 | 24% |
| Real box, badly placed (IoU 0.2 to 0.5) or a borderline unlabelled carton | 6 | 9% |
| Appliance: printer, coffee maker | 4 | 6% |
| Furniture: cube shelf, plywood panel, shelving | 4 | 6% |

Reading: fewer than half of the phantoms are "not a box" errors on a
solid object. The single biggest source is the detector firing several
times on one box. The prompt "cardboard box" is also loose about material:
anything rectangular with a lid, and some things without, score as a box.

## False negatives: 23 missed boxes

| category | count | share |
|---|---|---|
| Occluded, clutter-embedded, or atypical view (under a tote, behind a backpack, contents spilling, a closed box seen flat from above) | 9 | 39% |
| White or printed carton (bankers box, white file box, lamp box) | 8 | 35% |
| Motion blur | 6 | 26% |

## Worst condition

Blur, by a wide margin: recall 0.14 (1 of 7) versus 0.72 on sharp images.
Second is white boxes: recall 0.44 versus 0.67 brown. Occlusion costs
about 0.19 recall (0.63 occluded vs 0.82 unoccluded) and most of the
precision loss: 57 of the 68 phantoms are in occluded or cluttered scenes.
The 13 no-box images produced one phantom in total, so phantoms come from
clutter, not from empty scenes.

## Annotated examples

| image | type | what happened | hypothesis |
|---|---|---|---|
| `img_1799.jpg` | FN x3 | Three boxes, zero detections. | Motion blur smears every edge; with no corner or tape-line structure left, there is nothing box-like to anchor on. Blur removes exactly the high-frequency cues the model relies on. |
| `img_1813.jpg` | FN x2 | Open box on a chair plus one behind, nothing detected. | Same as above. Even the large, frame-filling box is lost once edges go soft. |
| `img_1723.jpg` | FN | Large closed box, flat on the floor, sharp, unoccluded, missed entirely. | Top-down view shows one flat face with no visible depth, and a hard shadow splits it in two. The model seems to need a 3-D corner or an open top to call it a box. |
| `img_1757.jpg` | FN x2, FP x2 | Two white bankers boxes missed; the foam cooler on top and the plywood panel detected instead. | "Cardboard box" is read as brown and corrugated. White printed cartons do not match that prior, while the cooler is box-shaped enough to pass. |
| `img_1733.jpg` | FN, FP x2 | Same scene, other angle. Bankers box missed, cooler and plywood detected. | Confirms it is the object, not the angle. The cooler is a consistent false positive across every image it appears in. |
| `img_1726.jpg` | FN, FP x2 | Box under a backpack missed; the small white carton on the table detected but far off (IoU 0.08). | Occlusion removes the top edges and the backpack breaks the silhouette. Occluded boxes need context the zero-shot model does not have. |
| `img_1783.jpg` | FP | Wide room shot. Both distant boxes found, plus a phantom on a white cube shelf. | Cube shelving has the same rectangular-cavity look as an open box. |
| `img_1708.jpg` | FP x4 | Both boxes found, plus a plastic tote, a printer, a dark corner and a white bin. | Material-blind prompt. Anything with straight edges and a lid is a candidate. |
| `img_1722.jpg` | FP x5 | One shipping box, six predictions. | Each flap and face of an open box gets its own detection. NMS does not merge them because the fragments barely overlap each other. |
| `img_1744.jpg` | FN, FP x5 | ACE box found once plus four fragment boxes and a bag; the open box with the helmet missed. | Fragment firing again, plus an open box whose contents (helmet, cables) replace the cardboard texture the model wants. |

## What this points to for the improvement

Fine-tuning on the train split should help the two largest buckets directly:
it teaches the model what *this* mix of brown and white boxes looks like,
and that totes, coolers and toolboxes are negatives. Blur augmentation
(motion blur, brightness) is the targeted change for the worst condition.
Duplicate detections may improve with training too, since the model learns
the full-box extent instead of flaps.
