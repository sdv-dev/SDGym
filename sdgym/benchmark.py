import json
import logging
import os
import urllib

import numpy as np
import pandas as pd

from sdgym.constants import CATEGORICAL, ORDINAL
from sdgym.evaluate import evaluate
from sdgym.synthesizers import BaseSynthesizer

LOGGER = logging.getLogger(__name__)


DATASETS = {
    'intrusion': {
        'categorical_columns': [1, 2, 3, 6, 11, 13, 19, 20, 40],
        'ordinals': [7, 8, 10, 14, 17, 18],
        'label': 40,
        'problem_type': 'multiclass_classification'
    },
    'news': {
        'categorical_columns': [11, 12, 13, 14, 15, 16, 29, 30, 31, 32, 33, 34, 35, 36],
        'ordinals': [],
        'label': 58,
        'problem_type': 'regression'
    },
    'adult': {
        'categorical_columns': [1, 5, 6, 7, 8, 9, 13, 14],
        'ordinals': [3],
        'label': 14,
        'problem_type': 'binary_classification'
    },
    'covtype': {
        'categorical_columns': [
            10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
            31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51,
            52, 53, 54
        ],
        'ordinals': [],
        'label': 54,
        'problem_type': 'multiclass_classification'
    },
    'credit': {
        'categorical_columns': [29],
        'ordinals': [],
        'label': 29,
        'problem_type': 'binary_classification'
    },
    'census': {
        'categorical_columns': [
            1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 19, 20, 21, 22, 23, 24, 25, 26, 27,
            28, 30, 31, 32, 33, 34, 35, 36, 37, 39, 40
        ],
        'ordinals': [],
        'label': 40,
        'problem_type': 'binary_classification'
    },
    'mnist12': {
        'categorical_columns': [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
            45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65,
            66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86,
            87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106,
            107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123,
            124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140,
            141, 142, 143, 144
        ],
        'ordinals': [],
        'label': 144,
        'problem_type': 'multiclass_classification'
    },
    'mnist28': {
        'categorical_columns': [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
            45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65,
            66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86,
            87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105,
            106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122,
            123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139,
            140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156,
            157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173,
            174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190,
            191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207,
            208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224,
            225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241,
            242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258,
            259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275,
            276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292,
            293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309,
            310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326,
            327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343,
            344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360,
            361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377,
            378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394,
            395, 396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411,
            412, 413, 414, 415, 416, 417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428,
            429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445,
            446, 447, 448, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462,
            463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479,
            480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491, 492, 493, 494, 495, 496,
            497, 498, 499, 500, 501, 502, 503, 504, 505, 506, 507, 508, 509, 510, 511, 512, 513,
            514, 515, 516, 517, 518, 519, 520, 521, 522, 523, 524, 525, 526, 527, 528, 529, 530,
            531, 532, 533, 534, 535, 536, 537, 538, 539, 540, 541, 542, 543, 544, 545, 546, 547,
            548, 549, 550, 551, 552, 553, 554, 555, 556, 557, 558, 559, 560, 561, 562, 563, 564,
            565, 566, 567, 568, 569, 570, 571, 572, 573, 574, 575, 576, 577, 578, 579, 580, 581,
            582, 583, 584, 585, 586, 587, 588, 589, 590, 591, 592, 593, 594, 595, 596, 597, 598,
            599, 600, 601, 602, 603, 604, 605, 606, 607, 608, 609, 610, 611, 612, 613, 614, 615,
            616, 617, 618, 619, 620, 621, 622, 623, 624, 625, 626, 627, 628, 629, 630, 631, 632,
            633, 634, 635, 636, 637, 638, 639, 640, 641, 642, 643, 644, 645, 646, 647, 648, 649,
            650, 651, 652, 653, 654, 655, 656, 657, 658, 659, 660, 661, 662, 663, 664, 665, 666,
            667, 668, 669, 670, 671, 672, 673, 674, 675, 676, 677, 678, 679, 680, 681, 682, 683,
            684, 685, 686, 687, 688, 689, 690, 691, 692, 693, 694, 695, 696, 697, 698, 699, 700,
            701, 702, 703, 704, 705, 706, 707, 708, 709, 710, 711, 712, 713, 714, 715, 716, 717,
            718, 719, 720, 721, 722, 723, 724, 725, 726, 727, 728, 729, 730, 731, 732, 733, 734,
            735, 736, 737, 738, 739, 740, 741, 742, 743, 744, 745, 746, 747, 748, 749, 750, 751,
            752, 753, 754, 755, 756, 757, 758, 759, 760, 761, 762, 763, 764, 765, 766, 767, 768,
            769, 770, 771, 772, 773, 774, 775, 776, 777, 778, 779, 780, 781, 782, 783, 784
        ],
        'ordinals': [],
        'label': 784,
        'problem_type': 'multiclass_classification'
    },
    'gridr': {
        'categorical_columns': [],
        'ordinals': [],
        'problem_type': 'gaussian_likelihood'
    },
    'asia': {
        'categorical_columns': [0, 1, 2, 3, 4, 5, 6, 7],
        'ordinals': [],
        'problem_type': 'bayesian_likelihood'
    },
    'child': {
        'categorical_columns': [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19
        ],
        'ordinals': [],
        'problem_type': 'bayesian_likelihood'
    },
    'grid': {
        'categorical_columns': [],
        'ordinals': [],
        'problem_type': 'gaussian_likelihood'
    },
    'insurance': {
        'categorical_columns': [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
            23, 24, 25, 26
        ],
        'ordinals': [],
        'problem_type': 'bayesian_likelihood'
    },
    'ring': {
        'categorical_columns': [],
        'ordinals': [],
        'problem_type': 'gaussian_likelihood'
    },
    'alarm': {
        'categorical_columns': [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36
        ],
        'ordinals': [],
        'problem_type': 'bayesian_likelihood'
    },
}


BASE_URL = 'http://sdgym.s3.amazonaws.com/datasets/'
DATA_PATH = os.path.join(os.path.dirname(__file__), 'data')


def _load_json(path):
    with open(path) as json_file:
        return json.load(json_file)


def _load_file(filename, loader):
    local_path = os.path.join(DATA_PATH, filename)
    if not os.path.exists(local_path):
        os.makedirs(DATA_PATH, exist_ok=True)
        urllib.request.urlretrieve(BASE_URL + filename, local_path)

    return loader(local_path)


def _load_dataset(name):
    data = _load_file(name + '.npz', np.load)
    meta = _load_file(name + '.json', _load_json)

    return data['train'], data['test'], meta


def _get_columns(metadata):
    categorical_columns = list()
    ordinal_columns = list()
    for column_idx, column in enumerate(metadata['columns']):
        if column['type'] == CATEGORICAL:
            categorical_columns.append(column_idx)
        elif column['type'] == ORDINAL:
            ordinal_columns.append(column_idx)

    return categorical_columns, ordinal_columns


def benchmark(synthesizer, datasets=DATASETS.keys(), repeat=3):
    if issubclass(synthesizer, BaseSynthesizer):
        synthesizer = synthesizer.run

    results = list()
    for name in datasets:
        LOGGER.info('Evaluating dataset %s', name)
        train, test, metadata = _load_dataset(name)

        categorical_columns, ordinal_columns = _get_columns(metadata)

        for iteration in range(repeat):
            synthesized = synthesizer(train, categorical_columns, ordinal_columns)
            scores = evaluate(train, test, synthesized, metadata)
            scores['dataset'] = name
            scores['iter'] = iteration
            results.append(scores)

    return pd.concat(results)
