import json
import rospy
import time
import cv2
import os
import shutil
import requests
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
import joblib 
# from sklearn.externals import joblib

from base_interface import MoveGroupPythonInterface
from utils import get_timestamp, get_cur_dir, get_datestamp
from gen_random_objects import idx_zh_map, zh_idx_map


ACTION_MODE_SPACE = {
    # 'd', 'move_delta',
    'pi', 'pick',
    'pl', 'place',
    'u', 'undo',
    'g', 'gear_move',
    'r', 'repeat'
}

# init_end_box_view1 = [322, 1, 713, 261] 
init_end_box_view1 = [322, 1, 391, 260]
# init_end_box_view2 = [230, 0, 300, 64]
init_end_box_view2 = [230, 0, 70, 64]

import datetime
def get_timestamp():
    return datetime.datetime.now().strftime('%Y%m%d%H%M%S')

def calculate_iou(bbox1, bbox2):  
  
    x1, y1, x2, y2 = bbox1  
    x1_prime, y1_prime, x2_prime, y2_prime = bbox2  
  
   
    x1_inter = max(x1, x1_prime)  
    y1_inter = max(y1, y1_prime)  
    x2_inter = min(x2, x2_prime)  
    y2_inter = min(y2, y2_prime)  
  
   
    area_inter = max(0, x2_inter - x1_inter + 1) * max(0, y2_inter - y1_inter + 1)  
  
    
    area_bbox1 = (x2 - x1 + 1) * (y2 - y1 + 1)  
    area_bbox2 = (x2_prime - x1_prime + 1) * (y2_prime - y1_prime + 1)  
  
     
    area_union = area_bbox1 + area_bbox2 - area_inter  
  
     
    iou = area_inter / area_union  
  
    return iou  
    
class GraspAnythingInterface(MoveGroupPythonInterface):
    def __init__(self, config):
        self.config = config
        super().__init__(config)

    def _init_inference(self):
        # start pose
        self._move_to_start_pose(pose_config_key='grasp_start_pose')
        self._place()

        self.deno_url = "http://127.0.0.1:6180/get_bbox"
        self.seg_url = "http://127.0.0.1:6181/get_mask"
        self.policy_url = "http://127.0.0.1:6692/policy" 
        self.tracker_init_api_1 = "http://127.0.0.1:6186/init_tracker"
        self.tracking_api_1 = "http://127.0.0.1:6186/tracking"
        self.tracker_init_api_2 = "http://127.0.0.1:6187/init_tracker"
        self.tracking_api_2 = "http://127.0.0.1:6187/tracking"

        self.end_tracker_init_api_1 = "http://127.0.0.1:6184/init_tracker"
        self.end_tracking_api_1 = "http://127.0.0.1:6184/tracking"

        self.end_tracker_init_api_2 = "http://127.0.0.1:6185/init_tracker"
        self.end_tracking_api_2 = "http://127.0.0.1:6185/tracking"

        self.decision_tree = joblib.load('/home/v-chuhaojin/my_project/grasp_anything_CoRL/decision_tree_model_all.pkl')
        self.end_effector_mean = np.load('/home/v-chuhaojin/my_project/grasp_anything_CoRL/end_effector_mean_all.npy')
        self.end_effector_std = np.load('/home/v-chuhaojin/my_project/grasp_anything_CoRL/end_effector_std_all.npy')

        self.session_id = get_timestamp()

        self.is_close_flag = False
    
        prompt = input('Enter prompt (e.g., pen-apple):').split('-')
        self.prompt_1 = prompt[0]
        self.prompt_2 = prompt[1]

        self.image_buffer_root_dir = f'/Data/training_test/FINAL_GRASP_DATA/testing_data_record/{self.session_id}-{self.prompt_1.replace(" ", "_")}-{self.prompt_2.replace(" ", "_")}'
        self.image_0_dir = os.path.join(self.image_buffer_root_dir, 'images', 'view_0')
        self.image_1_dir = os.path.join(self.image_buffer_root_dir, 'images', 'view_1')
        self.mask_0_dir = os.path.join(self.image_buffer_root_dir, 'masks', 'view_0')
        self.mask_1_dir = os.path.join(self.image_buffer_root_dir, 'masks', 'view_1')

        self.seg_request_0_dir = os.path.join(self.image_buffer_root_dir, 'seg_request', 'view_0')
        self.seg_request_1_dir = os.path.join(self.image_buffer_root_dir, 'seg_request', 'view_1')

        self.pos_action_dir = os.path.join(self.image_buffer_root_dir, 'pos_action')


        os.makedirs(self.image_0_dir)
        os.makedirs(self.image_1_dir)
        os.makedirs(self.mask_0_dir)
        os.makedirs(self.mask_1_dir)

        os.makedirs(self.seg_request_0_dir)
        os.makedirs(self.seg_request_1_dir)

        os.makedirs(self.pos_action_dir)

        self.image_0_received = False
        self.image_0_path = ''
        self.image_1_received = False
        self.image_1_path = ''
        self.gripper_state = 1  # 1 for open 0 for closed

    def inference(self):
        def observe(image_id):
            _save_image(image_id, 0)
            rospy.sleep(0.1)

            _save_image(image_id, 1)
            rospy.sleep(0.1)
        
        def gen_mask(image_id):
            self.bbox_dict = {}
            # mask 0:
            if image_id == 0:
                data = {
                    "image_path": self.image_0_path, 
                    "text_prompt_1": self.prompt_1,
                    "text_prompt_2": self.prompt_2,
                    "save_bbox_image": True
                }
                r = requests.post(self.deno_url, data = data)
                resp = r.json()
                self.bbox_dict["boxes_1_view1"] = resp["boxes_1"][0]
                init_box = resp["boxes_1"][0]
                resp["boxes_1"] = json.dumps(resp["boxes_1"])
                resp["logit_1"] = json.dumps(resp["logit_1"])
                resp["boxes_2"] = json.dumps(resp["boxes_2"])
                resp["logit_2"] = json.dumps(resp["logit_2"])
                self.view_1_boxes_2 = resp["boxes_2"]
                self.view_1_logit_2 = resp["logit_2"]
                #### Init the tracker.
                W, H = init_box[2] - init_box[0], init_box[3] - init_box[1]
                init_box = [init_box[0], init_box[1], W, H]
                init_data = {
                    "image_path": self.image_0_path,
                    "init_bbox": json.dumps(init_box)
                }
                self.init_box_v1 = init_box
                r = requests.post(self.tracker_init_api_1, data = init_data)
                print("init the tracker:", r.json())

                #### init the tracker for end-effector in view 1.
                init_data = {
                    "image_path": self.image_0_path,
                    "init_bbox": json.dumps(init_end_box_view1)
                }
                r = requests.post(self.end_tracker_init_api_1, data = init_data)
                self.end_box_view1 = [
                    init_end_box_view1[0],
                    init_end_box_view1[1],
                    init_end_box_view1[0] + init_end_box_view1[2],
                    init_end_box_view1[1] + init_end_box_view1[3]
                ]
                self.bbox_dict["end_box_view1"] = self.end_box_view1
            else:
                timestamp = get_timestamp()
                ##### Call Tracking API.
                track_data = {
                    "image_path": self.image_0_path
                }
                r = requests.post(self.tracking_api_1, data = track_data)
                tk_bbox_1 = r.json()
                ##### convert the bbox (x, y, w, h) to (x1, y1, x2, y2)
                W, H = tk_bbox_1[2], tk_bbox_1[3]
                tk_bbox_1[2], tk_bbox_1[3] = tk_bbox_1[0] + W, tk_bbox_1[1] + H
                print("Tracking Bounding Box: ", tk_bbox_1)
                self.bbox_dict["boxes_1_view1"] = tk_bbox_1
                resp = {
                    "boxes_1": json.dumps([tk_bbox_1, ]),
                    "boxes_2": self.view_1_boxes_2,
                    "logit_1": json.dumps([1.0, ]),
                    "logit_2": self.view_1_logit_2,
                    "image_path": self.image_0_path, 
                    "text_prompt_1": self.prompt_1,
                    "text_prompt_2": self.prompt_2,
                    "timestamp": timestamp,
                    "pred_img_path": "",
                }
                #### Call tracking API for end-effector in view1.
                track_data = {
                    "image_path": self.image_0_path
                }
                r = requests.post(self.end_tracking_api_1, data = track_data)
                boxes_end = r.json()
                self.end_box_view1 = [
                    boxes_end[0],
                    boxes_end[1],
                    boxes_end[0] + boxes_end[2],
                    boxes_end[1] + boxes_end[3]
                ]
                self.bbox_dict["end_box_view1"] = self.end_box_view1

            resp_2 = requests.post(self.seg_url, data = resp)
            mask_img_path = resp_2.text
            mask_0_path = os.path.join(self.mask_0_dir, f'{image_id}.jpg')
            shutil.copy(mask_img_path, mask_0_path)
            f = open(self.seg_request_0_dir + f'/{image_id}.json', "w")
            json.dump(resp, f)
            f.close()

            # mask 1
            if image_id == 0:
                data = {
                    "image_path": self.image_1_path, 
                    "text_prompt_1": self.prompt_1,
                    "text_prompt_2": self.prompt_2,
                    "save_bbox_image": True
                }
                r = requests.post(self.deno_url, data = data)
                resp = r.json()
                init_box = resp["boxes_1"][0]
                self.bbox_dict["boxes_1_view2"] = resp["boxes_1"][0]
                resp["boxes_1"] = json.dumps(resp["boxes_1"])
                resp["logit_1"] = json.dumps(resp["logit_1"])
                resp["boxes_2"] = json.dumps(resp["boxes_2"])
                resp["logit_2"] = json.dumps(resp["logit_2"])
                self.view_2_boxes_2 = resp["boxes_2"]
                self.view_2_logit_2 = resp["logit_2"]
                #### Init the tracker.
                W, H = init_box[2] - init_box[0], init_box[3] - init_box[1]
                init_box = [init_box[0], init_box[1], W, H]
                init_data = {
                    "image_path": self.image_1_path,
                    "init_bbox": json.dumps(init_box)
                }
                self.init_box_v2 = init_box
                r = requests.post(self.tracker_init_api_2, data = init_data)
                print("init the tracker:", r.json())

                #### init the tracker for end-effector in view 1.
                init_data = {
                    "image_path": self.image_1_path,
                    "init_bbox": json.dumps(init_end_box_view2)
                }
                r = requests.post(self.end_tracker_init_api_2, data = init_data)
                self.end_box_view2 = [
                    init_end_box_view2[0],
                    init_end_box_view2[1],
                    init_end_box_view2[0] + init_end_box_view2[2],
                    init_end_box_view2[1] + init_end_box_view2[3]
                ]
                self.bbox_dict["end_box_view2"] = self.end_box_view2
            else:
                timestamp = get_timestamp()
                ##### Call Tracking API.
                track_data = {
                    "image_path": self.image_1_path
                }
                r = requests.post(self.tracking_api_2, data = track_data)
                tk_bbox_1 = r.json()
                ##### convert the bbox (x, y, w, h) to (x1, y1, x2, y2)
                W, H = tk_bbox_1[2], tk_bbox_1[3]
                tk_bbox_1[2], tk_bbox_1[3] = tk_bbox_1[0] + W, tk_bbox_1[1] + H
                print("Tracking Bounding Box: ", tk_bbox_1)
                self.bbox_dict["boxes_1_view2"] = tk_bbox_1
                resp = {
                    "boxes_1": json.dumps([tk_bbox_1, ]),
                    "boxes_2": self.view_2_boxes_2,
                    "logit_1": json.dumps([1.0, ]),
                    "logit_2": self.view_2_logit_2,
                    "image_path": self.image_1_path, 
                    "text_prompt_1": self.prompt_1,
                    "text_prompt_2": self.prompt_2,
                    "timestamp": timestamp,
                    "pred_img_path": "",
                }
                #### Call tracking API for end-effector in view1.
                track_data = {
                    "image_path": self.image_1_path
                }
                r = requests.post(self.end_tracking_api_2, data = track_data)
                boxes_end = r.json()
                self.end_box_view2 = [
                    boxes_end[0],
                    boxes_end[1],
                    boxes_end[0] + boxes_end[2],
                    boxes_end[1] + boxes_end[3]
                ]
                self.bbox_dict["end_box_view2"] = self.end_box_view2
            resp_2 = requests.post(self.seg_url, data=resp)
            mask_img_path = resp_2.text
            mask_1_path = os.path.join(self.mask_1_dir, f'{image_id}.jpg')
            shutil.copy(mask_img_path, mask_1_path)

            f = open(self.seg_request_1_dir + f'/{image_id}.json', "w")
            json.dump(resp, f)
            f.close()

        
        def _get_robot_state():
            wpose = self.move_group.get_current_pose().pose
            position = wpose.position
            gripper_state = self.gripper_state

            joint_values = self.move_group.get_current_joint_values()

            return {
                'end_effector_state': [position.x, position.y, position.z, gripper_state],
                'joint_values': joint_values
            }
        
        def policy(image_id, state):
            image_0_path = os.path.join(self.image_0_dir, f'{image_id}.jpeg')
            mask_0_path = os.path.join(self.mask_0_dir, f'{image_id}.jpg')

            image_1_path = os.path.join(self.image_1_dir, f'{image_id}.jpeg')
            mask_1_path = os.path.join(self.mask_1_dir, f'{image_id}.jpg')

            data = {
                "image_0_path": image_0_path, 
                "mask_0_path": mask_0_path, 
                "image_1_path": image_1_path,
                "mask_1_path": mask_1_path,
                "image_id": image_id,
                "image_0_dir": self.image_0_dir,
                "mask_0_dir": self.mask_0_dir,
                "image_1_dir": self.image_1_dir,
                "mask_1_dir": self.mask_1_dir,
                "state": json.dumps(state)
            }
            r = requests.post(self.policy_url, data=data)
            resp = r.json()

            iou_view1 = calculate_iou(self.bbox_dict["boxes_1_view1"], self.bbox_dict["end_box_view1"])
            iou_view2 = calculate_iou(self.bbox_dict["boxes_1_view2"], self.bbox_dict["end_box_view2"])
            
            z_vaule = state["end_effector_state"][2]
            iou_z_feat = np.array([z_vaule, iou_view1, iou_view2])

            end_effector_feat = np.concatenate([
                self.bbox_dict["boxes_1_view1"], 
                self.bbox_dict["end_box_view1"], 
                self.bbox_dict["boxes_1_view2"], 
                self.bbox_dict["end_box_view2"], 
                iou_z_feat], axis = 0)[None, :]
            end_effector_feat = (end_effector_feat - self.end_effector_mean[None, :]) / (self.end_effector_std[None, :] + 0.00000001)
            print("end_effector_feat:", end_effector_feat)
            if z_vaule < 0.32:
                y_pred = self.decision_tree.predict(end_effector_feat)
            else:
                y_pred = [[1,], ]
            print("End effector predicted by decision tree:", y_pred[0])
            return resp['action'], resp['binary_class'], y_pred[0]

        def _save_image(image_id, view):
            # start subsribing
            if view == 0:
                print(f'Index: {image_id}')
                self.image_0_path = os.path.join(self.image_0_dir, f'{image_id}.jpeg')
                rospy.Subscriber(self.config['topics']['view_0'], Image, save_image_0_callback)
                start_time = time.time()
                # Wait for image message for 2 seconds
                # If signal received, callback function would be called and the loop ends
                while not self.image_0_received and time.time() - start_time < 2:
                    rospy.sleep(0.1)

                if not self.image_0_received:
                    raise RuntimeError('No image received from topic /rgb/image_raw')
                
                # reset image_received signal
                self.image_0_received = False

            elif view == 1:
                self.image_1_path = os.path.join(self.image_1_dir, f'{image_id}.jpeg')
                rospy.Subscriber(self.config['topics']['view_1'], Image, save_image_1_callback)
                start_time = time.time()
                # Wait for image message for 2 seconds
                # If signal received, callback function would be called and the loop ends
                while not self.image_1_received and time.time() - start_time < 2:
                    rospy.sleep(0.1)

                if not self.image_1_received:
                    raise RuntimeError('No image received from topic /rgb/image_raw')
                
                # reset image_received signal
                self.image_1_received = False

        def save_image_0_callback(msg):
            # mutex-like signal
            if self.image_0_received:
                return
            # Convert ROS Image message to OpenCV2
            cv2_img = bridge.imgmsg_to_cv2(msg, "bgr8")
            # Save OpenCV2 image as a jpeg 
            cv2.imwrite(self.image_0_path, self._crop_image(cv2_img, view=0))
            # print(f'image saved at {self.image_path}')
            self.image_0_received = True

        def save_image_1_callback(msg):
            # mutex-like signal
            if self.image_1_received:
                return
            # Convert ROS Image message to OpenCV2
            cv2_img = bridge.imgmsg_to_cv2(msg, "bgr8")
            # cv2_img = cv2.flip(cv2.transpose(cv2_img), 1)  # rotate 90 degrees clockwise
            # Save OpenCV2 image as a jpeg 
            cv2.imwrite(self.image_1_path, self._crop_image(cv2_img, view=1))
            # print(f'image saved at {self.image_path}')
            self.image_1_received = True

        self._init_inference()
        print('GraspAnything inferencing...')
        bridge = CvBridge()

        image_id = 0
        finish_flag = False
        place_threshold = 0
        past_high_flag = False
        pick_threshold = 0
        while True:
            observe(image_id)
            gen_mask(image_id)
            state = _get_robot_state()
            if finish_flag:
                pos_action = {}
                pos_action["state"] = state
                pos_action["action"] = [0., 0., 0.]
                pos_action["is_close_flag"] = self.is_close_flag
                f = open(os.path.join(self.pos_action_dir, f'{image_id}.json'), "w")
                json.dump(pos_action, f)
                f.close()
                break
            print("state:", state)
            action, binary_class, end_effector_close = policy(image_id, state)

            pos_action = {}
            pos_action["state"] = state
            pos_action["action"] = action

            print(f'action: {action}, binary_class: {binary_class}')
            
            # binary_class[0] > binary_class[1], open
            # if binary_class[0] > binary_class[1]:
            if end_effector_close == 0:
                gripper_state = 0
                #Iaction[2] = action[2] + 0.015
            else:
                gripper_state = 1
            
            if self.is_close_flag:
                if binary_class[1] > binary_class[0] - 2:
                    gripper_state = 1
                    finish_flag = True
                else:
                    gripper_state = 0
            # gripper_state = 1
            self.gripper_state = gripper_state
            now_z_pos = state["end_effector_state"][2]
            min_z = 0.271
            if self.is_close_flag:
                if now_z_pos > 0.3:
                    past_high_flag = True
                min_z = min_z_after_pick
                if now_z_pos < (min_z + 0.003) and past_high_flag:
                    place_threshold += 1
            else:
                min_z = 0.271
            if place_threshold == 4:
                gripper_state = 1
                finish_flag = True
            
            if now_z_pos < 0.282:
                if not self.is_close_flag:
                    pick_threshold += 1
            if pick_threshold == 2:
                gripper_state = 0
                pick_threshold = 0

            action[2] = max(action[2], min_z - now_z_pos)
            self._delta_pose(action[0], action[1], action[2])
            if gripper_state == 0:
                if not self.is_close_flag:
                    print("pick")
                    self._pick()
                    time.sleep(1)
                    state = _get_robot_state()
                    now_z_pos = state["end_effector_state"][2]
                    min_z_after_pick = now_z_pos
                    self.is_close_flag = True
            else:
                if self.is_close_flag:
                    print("_place")
                    self._place()
                    self.is_close_flag = False
            
            pos_action["is_close_flag"] = self.is_close_flag
            f = open(os.path.join(self.pos_action_dir, f'{image_id}.json'), "w")
            json.dump(pos_action, f)
            f.close()
            image_id += 1
            if finish_flag:
                time.sleep(4)
                observe(image_id)
                gen_mask(image_id)
                break


def main():
    config = json.load(open(os.path.join(get_cur_dir(), 'config.json'), 'r'))
    interface = GraspAnythingInterface(config)
    interface.inference()


if __name__ == '__main__':
    main()