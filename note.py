import ast
from argparse import ArgumentParser
from mmengine.logging import print_log
from mmdet.apis import DetInferencer
from mmdet.evaluation import get_classes
import cv2
import numpy as np

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('inputs', type=str, help='Input image file or folder path.')
    parser.add_argument('model', type=str, help='Config or checkpoint .pth file or the model name and alias defined in metafile.')
    parser.add_argument('--weights', default=None, help='Checkpoint file')
    parser.add_argument('--out-dir', type=str, default='outputs', help='Output directory of images or prediction results.')
    parser.add_argument('--texts', help='Text prompt, such as "bench . car .", "$: coco"')
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    parser.add_argument('--pred-score-thr', type=float, default=0.3, help='BBox score threshold')
    parser.add_argument('--batch-size', type=int, default=1, help='Inference batch size.')
    parser.add_argument('--show', action='store_true', help='Display the image in a popup window.')
    parser.add_argument('--no-save-vis', action='store_true', help='Do not save detection vis results')
    parser.add_argument('--no-save-pred', action='store_true', help='Do not save detection json results')
    parser.add_argument('--print-result', action='store_true', help='Whether to print the results.')
    parser.add_argument('--palette', default='none', choices=['coco', 'voc', 'citys', 'random', 'none'], help='Color palette used for visualization')
    parser.add_argument('--custom-entities', '-c', action='store_true', help='Whether to customize entity names?')
    parser.add_argument('--chunked-size', '-s', type=int, default=-1, help='If the number of categories is very large, you can specify this parameter to truncate multiple predictions.')
    parser.add_argument('--tokens-positive', '-p', type=str, help='Used to specify which locations in the input text are of interest.')
    call_args = vars(parser.parse_args())

    if call_args['no_save_vis'] and call_args['no_save_pred']:
        call_args['out_dir'] = ''

    if call_args['model'].endswith('.pth'):
        print_log('The model is a weight file, automatically assign the model to --weights')
        call_args['weights'] = call_args['model']
        call_args['model'] = None

    if call_args['texts'] is not None:
        if call_args['texts'].startswith('$:'):
            dataset_name = call_args['texts'][3:].strip()
            class_names = get_classes(dataset_name)
            call_args['texts'] = [tuple(class_names)]

    if call_args['tokens_positive'] is not None:
        call_args['tokens_positive'] = ast.literal_eval(call_args['tokens_positive'])

    init_kws = ['model', 'weights', 'device', 'palette']
    init_args = {}
    for init_kw in init_kws:
        init_args[init_kw] = call_args.pop(init_kw)

    return init_args, call_args

def main():
    init_args, call_args = parse_args()
    inferencer = DetInferencer(**init_args)
    chunked_size = call_args.pop('chunked_size')
    inferencer.model.test_cfg.chunked_size = chunked_size

    # Run inference
    results = inferencer(**call_args)

    # Extract the bounding boxes (bboxes) and labels
    bboxes = results['predictions'][0]['bboxes']
    labels = results['predictions'][0]['labels']
    scores = results['predictions'][0]['scores']

    # Convert bboxes and scores to numpy arrays for NMS
    bboxes = np.array(bboxes)
    scores = np.array(scores)

    # Apply NMS to remove redundant boxes but keep multiple boxes if they are not overlapping too much
    nms_threshold = 0.5  # You can adjust this to keep more boxes
    indices = cv2.dnn.NMSBoxes(bboxes.tolist(), scores.tolist(), score_threshold=call_args['pred_score_thr'], nms_threshold=nms_threshold)

    if len(indices) > 0:
        print("Bounding Boxes after NMS:")
        for idx in indices.flatten():
            final_bbox = bboxes[idx]
            print(f"Box {idx}: {', '.join(map(str, final_bbox))}")
    else:
        print("No bounding box after NMS")

    # Save visualized image
    visualized_image = results['visualization'][0]
    cv2.imwrite(f"{call_args['out_dir']}/annotated_image.jpg", visualized_image)

    if call_args['show']:
        cv2.imshow("Result", visualized_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if call_args['out_dir'] != '' and not (call_args['no_save_vis'] and call_args['no_save_pred']):
        print_log(f'results have been saved at {call_args["out_dir"]}')

if __name__ == '__main__':
    main()
