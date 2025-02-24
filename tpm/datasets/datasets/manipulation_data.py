import os
from PIL import Image
import webdataset as wds
from tpm.datasets.datasets.base_dataset import BaseDataset
from tpm.datasets.datasets.caption_datasets import CaptionDataset

import PIL
from torchvision import transforms

import json


class CCSBUDataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, location):
        super().__init__(vis_processor=vis_processor, text_processor=text_processor)

        print("ccsbu1")
        print(5)

    def to_dict(self, sample):
        return {
            "image": sample[0],
            "text_input": self.text_processor(sample[1]["caption"]),
        }




import os
import numpy as np
from PIL import Image
class Manipulation_data(BaseDataset):
    def __init__(self, vis_processor, text_processor, vis_root):
        """
        vis_root (string): Root directory of images (e.g. /Data/grasp_data/)
        """
        super().__init__(vis_processor, text_processor, vis_root)
        # todo
        #vis_root = '/Data/wenhui_grasp_data'
        self.dataset_folders = [os.path.join(vis_root, folder) for folder in os.listdir(vis_root)]
        self.annotation = []



        for folder in self.dataset_folders:
            images_path = os.path.join(folder, 'images')

            #actions_file = os.path.join(folder, 'data.npy')
            #actions = np.load(actions_file).astype(np.float32)


            actions_file = os.path.join(folder, 'data.json')


            with open(actions_file, 'r', encoding='utf-8') as file:
                data = json.load(file)

            #data = json.loads(actions_file)
            target_action = [0.0, 0.0, 0.0, 0]
            for item in data:
                action = item['action']
                if action == target_action:
                    end_effector_state1 = np.array(item['state']['end_effector_state'])
                    break
            end_effector_state2 = np.array(data[-1]['state']['end_effector_state'])

            actions = []
            states = []
            joint_values = []
            for item in data:
                action = item['action']
                actions.append(action)
                state = item['state']['end_effector_state']
                states.append(state)

                joint_values.append(item['state']['joint_values'])

            actions = np.array(actions)
            states = np.array(states)

            joint_values = np.array(joint_values)


            kkk = 0
            jjj = len(actions)
            for kkk in range(0, 100):
                if kkk < len(actions):  # 添加这个条件以确保不会访问越界的元素
                    if actions[kkk][3] == 0.0:
                        break
                else:  # 当kkk超过actions的行数时，跳出循环
                    break



            # Resampling
            for i, action in enumerate(actions):
                if(kkk == i):
                    weight = 4.0
                elif(((kkk - i) > 0) and ((kkk - i)<=2)):
                    weight = 4.0
                    #weight = 1.0
                elif (((kkk - i) == -1)):
                    weight = 4.0
                    #weight = 1.0
                elif (((kkk - i) < -1) and ((i - kkk) <= 2)):
                    weight = 4.0
                    #weight = 1.0
                elif ((jjj - i) == 1):
                    weight = 7.0
                # elif ((jjj - i) == 2):
                #     weight = 4.0
                elif ((jjj - i) >= 2  and (jjj - i) <= 5):
                    weight = 5.0
                else:
                    weight = 1.0

                state = states[i]
                joint_value = joint_values[i]
                ann = {'image_id': i, 'folder': folder, 'action': action, 'weights': weight, 'state':state, 'position1':end_effector_state1, 'position2':end_effector_state2, 'joint_value':joint_value}

                self.annotation.append(ann)

        print("Resampling")
        result = []
        for item in self.annotation:
            result.append(item)
            if item['weights'] == 5.0:
                for _ in range(2):
                    result.append(item.copy())
            elif item['weights'] == 4.0:
                for _ in range(2):
                    result.append(item.copy())
            elif item['weights'] == 3.0:
                for _ in range(1):
                    result.append(item.copy())
            elif item['weights'] == 7.0:
                for _ in range(3):
                    result.append(item.copy())
        self.annotation = result
        print("finish")




    def __getitem__(self, index):

        import copy

   
        try:

            ann = self.annotation[index]
            folder = ann["folder"]

            image_0_path = os.path.join(folder, 'images', 'view_0', '{}.jpeg'.format(ann["image_id"]))
            image_1_path = os.path.join(folder, 'images', 'view_1', '{}.jpeg'.format(ann["image_id"]))

            image_0 = Image.open(image_0_path).convert("RGB")
            image_1 = Image.open(image_1_path).convert("RGB")

            vis_processor_copy = copy.deepcopy(self.vis_processor)

            # Remove color_jitter from vis_processor_copy
            new_transforms = [
                transform for transform in vis_processor_copy.transform.transforms
                if not isinstance(transform, transforms.ColorJitter)
            ]
            mask_transform = transforms.Compose(new_transforms)

            image_0 = self.vis_processor(image_0)
            #image_1 = self.vis_processor(image_1)
            image_1 = vis_processor_copy(image_1)


            # mask_0_path = os.path.join(folder, 'dino_seg_mask', 'view_0', '{}.jpeg'.format(ann["image_id"]))
            # mask_1_path = os.path.join(folder, 'dino_seg_mask', 'view_1', '{}.jpeg'.format(ann["image_id"]))
            mask_0_path = os.path.join(folder, 'dino_seg_mask_fix', 'view_0', '{}.jpeg'.format(ann["image_id"]))
            mask_1_path = os.path.join(folder, 'dino_seg_mask_fix', 'view_1', '{}.jpeg'.format(ann["image_id"]))

            mask_0 = Image.open(mask_0_path).convert("RGB")
            mask_1 = Image.open(mask_1_path).convert("RGB")

            # mask_0 = self.vis_processor(mask_0)
            # #mask_1 = self.vis_processor(mask_1)
            # mask_1 = vis_processor_copy(mask_1)

            mask_0 = mask_transform(mask_0)
            # mask_1 = self.vis_processor(mask_1)
            mask_1 = mask_transform(mask_1)


            #action = self.text_processor(ann["action"])

            action = ann["action"].astype(np.float32)
            weight = ann["weights"]

            state = ann["state"].astype(np.float32)
            position1 = ann["position1"].astype(np.float32)
            position2 = ann['position2'].astype(np.float32)

            # could replace with your state data
            mean_p = np.array([0.4478190983908197, -0.011622959152025521, 0.3698577584487156])
            std_p = np.array([0.07212793256181435, 0.11386907480887651, 0.0557807174382576])


            state[:3] = (state[:3] - mean_p) / std_p

            joint_value = ann['joint_value'].astype(np.float32)

            return {
                "image_view_0": image_0,
                "image_view_1": image_1,
                "mask_view_0": mask_0,
                "mask_view_1": mask_1,
                "action_xyz": action[0:3],
                "action_griper": action[3:4],
                "image_id": ann["image_id"],
                "weights": weight,

                "state": state,
                "position1": position1,
                "position2": position2,

                'joint_value': joint_value,
            }

        except FileNotFoundError as e:
            print(f"File not found: {e.filename}")
            #self.invalid_indices.append(valid_index)
            return self.__getitem__(index + 1)

        except PIL.UnidentifiedImageError as e:
            print('Cannot identify image file')
            return self.__getitem__(index + 1)

        except OSError as e:
            print(f'Caught OSError: {e}')
            return self.__getitem__(index + 1)
