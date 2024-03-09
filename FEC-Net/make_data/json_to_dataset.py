import argparse
import json
import os
import os.path as osp
import warnings
import imgviz
import PIL.Image
import yaml

from labelme import utils
import base64


def main():
    count = os.listdir("./before/")
    for i in range(0, len(count)):
        path = os.path.join("./before", count[i])

        # 找到before文件中以json为结尾的文件并且打开
        if os.path.isfile(path) and path.endswith('json'):
            data = json.load(open(path))

            if data['imageData']:
                imageData = data['imageData']
            else:
                imagePath = os.path.join(os.path.dirname(path), data['imagePath'])
                with open(imagePath, 'rb') as f:
                    imageData = f.read()
                    imageData = base64.b64encode(imageData).decode('utf-8')
            img = utils.img_b64_to_arr(imageData)
            label_name_to_value = {'_background_': 0}
            for shape in data['shapes']:
                label_name = shape['label']
                if label_name in label_name_to_value:
                    label_value = label_name_to_value[label_name]
                else:
                    label_value = len(label_name_to_value)
                    label_name_to_value[label_name] = label_value

            # label_values must be dense
            label_values, label_names = [], []
            for ln, lv in sorted(label_name_to_value.items(), key=lambda x: x[1]):
                label_values.append(lv)
                label_names.append(ln)
            assert label_values == list(range(len(label_values)))

            lbl, _ = utils.shapes_to_label(img.shape, data['shapes'], label_name_to_value)

            # captions = ['{}: {}'.format(lv, ln)
            #     for ln, lv in label_name_to_value.items()]
            # lbl_viz = utils.labelme_shapes_to_label(lbl, img, captions)
            label_names = [None] * (max(label_name_to_value.values()) + 1)
            for name, value in label_name_to_value.items():
                label_names[value] = name
            # 4
            # lbl_viz = imgviz.label2rgb(
            #     lbl, imgviz.asgray(img), label_names=label_names, loc="rb"
            # )

            out_dir = osp.basename(count[i]).replace('.', '_')
            out_dir = osp.join(osp.dirname(count[i]), out_dir)
            out_dir = osp.join("output", out_dir)

            if not osp.exists(out_dir):
                os.mkdir(out_dir)
            # image
            PIL.Image.fromarray(img).save(osp.join(out_dir, 'img.png'))
            # label.png
            utils.lblsave(osp.join(out_dir, 'label.png'), lbl)
            # PIL.Image.fromarray(lbl_viz).save(osp.join(out_dir, 'label_viz.png'))
            # label_viz.png
            # PIL.Image.fromarray(lbl_viz).save(osp.join(out_dir, "label_viz.png"))

            with open(osp.join(out_dir, 'label_names.txt'), 'w') as f:
                for lbl_name in label_names:
                    f.write(lbl_name + '\n')

            # warnings.warn('info.yaml is being replaced by label_names.txt')
            # info = dict(label_names=label_names)
            # with open(osp.join(out_dir, 'info.yaml'), 'w') as f:
            #     yaml.safe_dump(info, f, default_flow_style=False)

            print('Saved to: %s' % out_dir)


if __name__ == '__main__':
    main()

