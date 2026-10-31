"""Narration script: the single source of truth for the explainer video.
Each scene is a list of (beat_id, text). Every number is taken from paper/CLAIMS.md, the paper's tables or the run artifacts
named there; nothing here is new. tts.py turns each beat into audio, and the scene code times its animations to it."""

SCENES = {
    "S01_Problem": [
        ("b1", "A robot reaches into a bin of identical industrial parts. A spacecraft closes in on a target it has to dock with. "
               "Both need the same thing from a single image: the six-dimensional pose of a known object. A rotation R in S O three, "
               "and a translation t in R three."),
        ("b2", "Usually, every new object means fine-tuning a large network. Here, the large networks never change. Two frozen "
               "foundation models, DINO v3 for appearance and MoGe 2 for surface geometry, are shared by every application. "
               "Only a small head, about two million parameters, is trained for each object."),
        ("b3", "So the question of this paper is simple. How far can such small heads go, and what do they need to get there?"),
    ],
    "S02_Ambiguity": [
        ("b1", "The hard part is not representing the pose. It is choosing it. Take a cube. Turn it by ninety degrees about a face "
               "axis, and the picture does not change. Twenty-four rotations produce the same image, and every one of them is correct."),
        ("b2", "Now take an object that is only nearly symmetric. A half turn looks almost the same, but only one of the two answers "
               "is right, and under a change of lighting the small details that separate them are exactly what becomes unreliable."),
        ("b3", "Draw every rotation as a point in a ball of radius pi: the direction is the axis, the length is the angle. "
               "A predictor's belief is not a single bump but several. Symmetry creates modes that are all correct. Near symmetry "
               "creates look-alike modes that are wrong. A single prediction has to select one of them."),
    ],
    "S03_Quaternions": [
        ("b1", "To build a space of hypotheses, write a rotation as a unit quaternion, q equals w plus x i plus y j plus z k: "
               "a point on the three-sphere in four dimensions."),
        ("b2", "A quaternion and its negative describe the same rotation, so the sphere covers the rotations twice. "
               "Rotating a vector is a sandwich product, and the angle between two rotations is two times the arc cosine of the "
               "absolute value of their dot product."),
    ],
    "S04_Tesseract": [
        ("b1", "Here is the central construction. Surround the three-sphere by a four-dimensional cube, the tesseract. Each "
               "quaternion is assigned to the cube face where its largest coordinate lives. That index is its chart."),
        ("b2", "Multiply by the sign of that coordinate, so it is positive. This identifies q with minus q. Then project "
               "radially onto the face: x equals q divided by q c. Three free coordinates remain, each between minus one and one."),
        ("b3", "Now halve each coordinate. The cube splits into eight children, and three bits name the child: b zero, plus two "
               "b one, plus four b two."),
        ("b4", "Repeat five times. Four charts, times eight to the fifth, gives one hundred thirty-one thousand and seventy-two "
               "cells. A rotation has become a path: one chart, then five choices out of eight."),
        ("b5", "Each cell is represented by its centre, lifted back onto the sphere. Using centres rather than corners means every "
               "rotation has exactly one path."),
        ("b6", "Watch one rotation being decoded. After the chart and the first split, the cell centre is on average about forty "
               "degrees away. Every further level halves that error: twenty, ten, five, and two and a half. At the leaf, the mean "
               "quantisation error is two point five one degrees."),
    ],
    "S05_Features": [
        ("b1", "The image passes through DINO v3 once. Its last-layer patch tokens are averaged down to an eight by eight grid "
               "of vectors, each with one thousand and twenty-four numbers."),
        ("b2", "MoGe 2 predicts a surface normal for every pixel. A foreground mask on which both backbones agree keeps only the "
               "object, and the masked normals are pooled into sixteen by sixteen cells of four numbers each."),
        ("b3", "Neither backbone is ever trained. The features are computed once per image and cached, so training a head "
               "takes minutes."),
    ],
    "S06_TreeHead": [
        ("b1", "The head reads these tokens with one Transformer encoder layer and a small causal decoder. It starts from a root "
               "token and predicts the chart: four numbers."),
        ("b2", "Then the token of the current cell centre is appended, and the decoder predicts which of the eight children comes "
               "next. Six decisions take it from the root to a leaf."),
        ("b3", "Training uses soft targets. At each level, the children receive probability proportional to e to the minus "
               "delta over tau, squared, where delta is the angle from the true rotation to the child's centre. The width tau "
               "halves with the cell size, from thirty-nine degrees at the root to two and a half at the leaf."),
        ("b4", "The appearance tree and the geometry tree are fused without any training. At every step their log-probabilities "
               "are added, a product of experts, and the search continues on the fused distribution."),
    ],
    "S07_Symmetry": [
        ("b1", "For a symmetric object, the annotated rotation is just one member of a set: q times s, for every symmetry s. "
               "Training on that one label gives contradictory targets for identical images."),
        ("b2", "The usual fix replaces every label by a canonical member, for example the one closest to the identity. But watch "
               "the canonical label as the object turns. It jumps. Two nearly identical views get labels far apart, and the "
               "relation between rotated views is broken."),
        ("b3", "Instead, we train on all of them. The loss is the average of the tree's loss over every equivalent path q s, "
               "using up to eight sampled members when the set is larger. The tree learns to split its probability between "
               "the equivalent paths. And for an object without symmetry, this is exactly the original loss."),
    ],
    "S08_Roll": [
        ("b1", "Rotating the crop of an image is not just augmentation. It is exactly a rotation of the camera about the ray d "
               "through the centre of the crop."),
        ("b2", "So the labels transform exactly. The new rotation is Exp of minus theta d, times q, and the translation is "
               "rotated by the same camera roll."),
        ("b3", "For objects far from the optical axis, rotating the crop is not quite a camera roll, so the image is warped with "
               "the homography K R K inverse, which makes the relation exact."),
    ],
    "S09_Orbit": [
        ("b1", "This gives a test-time tool. Rotate the crop by zero, ninety, one hundred eighty and two hundred seventy degrees. "
               "In each view, beam search returns the tree's eight best leaves."),
        ("b2", "Map every candidate back through the inverse roll. Now thirty-two candidates live in the same frame."),
        ("b3", "Then score each candidate in every view: map it into that view, follow its path down the tree, and add up the "
               "log-probabilities. The score is the sum over the four views."),
        ("b4", "A look-alike flip can win in one view, but the other views find it unlikely. The true pose is consistent by "
               "construction, so the highest total picks it. The price is four passes instead of one."),
    ],
    "S10_Distill": [
        ("b1", "Orbit voting can also teach. For every training image, the orbit over its three training views produces "
               "twenty-four candidates with soft weights."),
        ("b2", "A new single-view student learns to match them, level by level, together with the ground truth. At test time "
               "it costs one view."),
    ],
    "S11_Translation": [
        ("b1", "Translation is read from the same tokens. One learned query attends to the encoded image, and a small network "
               "predicts three numbers, u."),
        ("b2", "u is defined relative to the crop box: the offset of the object's projected centre from the box centre, in box "
               "units, and a depth term that grows with the box size times the depth. It inverts in closed form. The whole "
               "translation block has a third of a million parameters."),
    ],
    "S12_Flat": [
        ("b1", "Why a tree at all? A flat classifier scores all one hundred thirty-one thousand anchors at once. Its output "
               "layer alone has thirty-three point six million parameters, more than ten times the whole tree."),
        ("b2", "A greedy decode of the tree visits only forty-four nodes, and the tree does not grow with resolution. Against a "
               "flat classifier over SPACE HOP's Hopf grid, trained the same way, the error is three point eight seven degrees "
               "instead of seven point five three."),
        ("b3", "One caveat. That grid has only twelve in-plane rolls, so its own quantisation floor is six degrees. A control on "
               "the tree's own grid is still running, so this comparison does not yet isolate the hierarchy."),
    ],
    "S13_General": [
        ("b1", "On T-LESS, textureless industrial parts, multi-path targets with orbit voting place seventy-nine percent of "
               "real test rotations within ten degrees, against seventy-three point five with a single label."),
        ("b2", "On SYMSOL-T, every symmetric solid improves, from two point six two to two point zero seven degrees on average. "
               "LiePose, a diffusion model with a fine-tuned backbone, is still about four times more accurate."),
        ("b3", "On YCB-V household objects, trees trained only on real videos fail on new sequences. With synthetic renders, "
               "the error drops to twelve point nine degrees. Here orbit voting hurts, so validation turns it off."),
        ("b4", "Translation from the frozen tokens places eighty-eight percent of T-LESS objects within five centimetres, "
               "close to LiePose, and reaches one point six centimetres on YCB-V. On T-LESS and YCB-V these rotations still "
               "use the ground-truth translation for the frame conversion; that evaluation is pending."),
    ],
    "S14_Spacecraft": [
        ("b1", "Now the application. Spacecraft pose, trained only on renders and tested on real images from two "
               "hardware-in-the-loop facilities."),
        ("b2", "The real errors are dominated by half-turn flips. Yet an oracle that picks the best of the tree's own candidates "
               "reaches nine point eight degrees, against twenty-six point nine for the prediction. The right mode is there. "
               "It is just not selected."),
        ("b3", "Augmentation brings the error to seventeen and twenty degrees, orbit voting to twelve, and distillation to "
               "eleven point two and eleven point one on the two real domains. Orbit voting repairs sixty-two and eighty percent "
               "of severe flips, and breaks only zero point four percent of good predictions."),
        ("b4", "With a detector trained on renders instead of ground-truth boxes, the error doubles on one domain. There, "
               "detection, not the heads, is the bottleneck."),
    ],
    "S15_FineTuning": [
        ("b1", "Fine-tuned methods are still more accurate, and the clearest evidence of why uses the same features: a head on "
               "frozen DINO v3 features reaches fifteen and twenty-four degrees; adapting the backbone with LoRA reaches about "
               "one point two and one point six."),
        ("b2", "But adaptation is expensive. Those methods back-propagate through networks of twelve to eight hundred and forty "
               "million parameters, one to seven million times per target. Our backbones receive no gradient at all."),
        ("b3", "On the same frozen features, the selection-aware heads close thirty to fifty-eight percent of that gap, and they "
               "match or beat fine-tuned direct regressors, while one backbone pass serves every object's heads."),
    ],
    "S16_Synthesis": [
        ("b1", "So, back to the robot and the spacecraft. The correct pose was usually among the hypotheses. The difficulty was "
               "choosing it."),
        ("b2", "A hierarchical partition of rotations makes the choice explicit. Multi-path targets handle exact symmetry at no "
               "cost. Orbit voting handles look-alikes where it helps, at four times the cost. And all of it runs on backbones "
               "that never change, with heads small enough to train for each new object in minutes."),
    ],
}
