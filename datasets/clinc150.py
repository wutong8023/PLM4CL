"""


Author: Tong
Time: 18-04-2021
"""
import os
import json
from abc import ABC

import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from utils.conf import base_path
from backbone import import_from
from backbone.utils.tokenize import CustomizedTokenizer
from backbone.PTMClassifier import PTMClassifier
from datasets.utils.nlp_dataset import NLPDataset
from datasets.utils.continual_dataset import ContinualDataset, store_masked_loaders, store_masked_loaders_for_online
from datasets.utils.download_googledrive import download_file_from_google_drive as download_file


class CLINC150(NLPDataset):
    def __init__(self, download=True, data_file="clinc150/data_full.json", mod="train", tokenizer=None, ptm='bert',
                 slice_size: int = 5, quick_load=False, filter_rate: float = 1):
        super(CLINC150, self).__init__()
        
        self.label_dic = {
            "banking": [
                "transfer",
                "transactions",
                "balance",
                "freeze_account",
                "pay_bill",
                "bill_balance",
                "bill_due",
                "interest_rate",
                "routing",
                "min_payment",
                "order_checks",
                "pin_change",
                "report_fraud",
                "account_blocked",
                "spending_history"
            ],
            "credit_cards": [
                "credit_score",
                "report_lost_card",
                "credit_limit",
                "rewards_balance",
                "new_card",
                "application_status",
                "card_declined",
                "international_fees",
                "apr",
                "redeem_rewards",
                "credit_limit_change",
                "damaged_card",
                "replacement_card_duration",
                "improve_credit_score",
                "expiration_date"
            ],
            "kitchen_&_dining": [
                "recipe",
                "restaurant_reviews",
                "calories",
                "nutrition_info",
                "restaurant_suggestion",
                "ingredients_list",
                "ingredient_substitution",
                "cook_time",
                "food_last",
                "meal_suggestion",
                "restaurant_reservation",
                "confirm_reservation",
                "how_busy",
                "cancel_reservation",
                "accept_reservations"
            ],
            "home": [
                "shopping_list",
                "shopping_list_update",
                "next_song",
                "play_music",
                "update_playlist",
                "todo_list",
                "todo_list_update",
                "calendar",
                "calendar_update",
                "what_song",
                "order",
                "order_status",
                "reminder",
                "reminder_update",
                "smart_home"
            ],
            "auto_&_commute": [
                "traffic",
                "directions",
                "gas",
                "gas_type",
                "distance",
                "current_location",
                "mpg",
                "oil_change_when",
                "oil_change_how",
                "jump_start",
                "uber",
                "schedule_maintenance",
                "last_maintenance",
                "tire_pressure",
                "tire_change"
            ],
            "travel": [
                "book_flight",
                "book_hotel",
                "car_rental",
                "travel_suggestion",
                "travel_alert",
                "travel_notification",
                "carry_on",
                "timezone",
                "vaccines",
                "translate",
                "flight_status",
                "international_visa",
                "lost_luggage",
                "plug_type",
                "exchange_rate"
            ],
            "utility": [
                "time",
                "alarm",
                "share_location",
                "find_phone",
                "weather",
                "text",
                "spelling",
                "make_call",
                "timer",
                "date",
                "calculator",
                "measurement_conversion",
                "flip_coin",
                "roll_dice",
                "definition"
            ],
            "work": [
                "direct_deposit",
                "pto_request",
                "taxes",
                "payday",
                "w2",
                "pto_balance",
                "pto_request_status",
                "next_holiday",
                "insurance_change",
                "schedule_meeting",
                "pto_used",
                "meeting_schedule",
                "rollover_401k",
                "income",
                "insurance"
            ],
            "small_talk": [
                "greeting",
                "goodbye",
                "tell_joke",
                "where_are_you_from",
                "how_old_are_you",
                "what_is_your_name",
                "who_made_you",
                "thank_you",
                "what_can_i_ask_you",
                "what_are_your_hobbies",
                "do_you_have_pets",
                "are_you_a_bot",
                "meaning_of_life",
                "who_do_you_work_for",
                "fun_fact"
            ],
            "meta": [
                "change_ai_name",
                "change_user_name",
                "cancel",
                "user_name",
                "reset_settings",
                "whisper_mode",
                "repeat",
                "no",
                "yes",
                "maybe",
                "change_language",
                "change_accent",
                "change_volume",
                "change_speed",
                "sync_device"
            ]
        }
        self.domains = self.label_dic.keys()
        self.labels = [label for domain in self.domains for label in self.label_dic[domain]]
        
        # self.shuffle_id = np.arange(len(self.labels))
        # np.random.shuffle(self.shuffle_id)
        # self.label2id = {label: self.shuffle_id[idx] for idx, label in enumerate(self.labels)}
        self.label2id = {label: idx for idx, label in enumerate(self.labels)}
        
        self.domain2id = {domain: idx for idx, domain in enumerate(self.domains)}
        self.label2domain = {label: domain for domain in self.domains for label in self.label_dic[domain]}
        
        self.filter_rate = filter_rate
        self.ptm = ptm

        if tokenizer is None:
            self.tokenizer = CustomizedTokenizer(ptm=self.ptm, max_len=36)
        else:
            self.tokenizer = tokenizer
        
        if not quick_load:
            # An Evaluation Dataset for Intent Classification and Out-of-Scope Prediction EMNLP 2019
            # https://drive.google.com/file/d/13-MkxdzWcW_FkYNyBcU_FUh1WFplA3Xq/view?usp=sharing
            if download and not os.path.exists(os.path.join(base_path(), data_file)):
                self.clinc150_id = '13-MkxdzWcW_FkYNyBcU_FUh1WFplA3Xq'
                download_file(self.clinc150_id, base_path(), file_name="clinc150")
            self.data_file = os.path.join(base_path(), data_file)
            
            # predefined dataloader for clinc150
            # process seq data
            self.train_data, self.valid_data, self.test_data = self._prepare_data()
            if self.filter_rate < 1:
                self.train_data = self._filter_data(self.train_data, self.filter_rate)
            # targets
            self.train_targets = np.array([self.label2id[item["y"]] for item in self.train_data])
            self.valid_targets = np.array([self.label2id[item["y"]] for item in self.valid_data])
            self.test_targets = np.array([self.label2id[item["y"]] for item in self.test_data])
            
            # process use mode
            if mod == "train":
                self.data = self.train_data
            elif mod == "test":
                self.data = self.test_data
            else:
                self.data = self.valid_data
            self.targets = np.array([self.label2id[item["y"]] for item in self.data])
            
            # process online data
            self.slice_size = slice_size
            self.raw_data = self.train_data + self.valid_data + self.test_data
            self.online_data, self.sliced_online_data_idx = self._process_online_data(self.raw_data, self.slice_size)
            
            # distribution
            self.distribution = self._stat_data(self.raw_data)
        else:
            self.data = None
            self.targets = None

    def _prepare_data(self):
        data_path = self.data_file
        with open(data_path, "r", encoding="utf-8") as data_in:
            data = json.load(data_in)
            train_data = data["train"]
            train_data = self._format_data(train_data)
            valid_data = data["val"]
            valid_data = self._format_data(valid_data)
            test_data = data["test"]
            test_data = self._format_data(test_data)

        return train_data, valid_data, test_data


class SequentialCLINC150(ContinualDataset):
    NAME = "seq-clinc150"
    SETTING = "class-il"
    N_CLASSES_PER_TASK = 10
    N_TASKS = 15
    
    def get_data_loaders(self):
        # get the based dataset
        if self.dataset is None:
            self.dataset = CLINC150(mod="train",  ptm=self.args.ptm, filter_rate=self.args.filter_rate)
        
        # get new dataset
        train_dataset = CLINC150(quick_load=True)
        train_dataset.set_data(self.dataset.train_data)
        
        if self.args.validation:
            test_dataset = CLINC150(quick_load=True)
            test_dataset.set_data(self.dataset.valid_data)
        else:
            test_dataset = CLINC150(quick_load=True)
            test_dataset.set_data(self.dataset.test_data)
        
        train, test = store_masked_loaders(train_dataset, test_dataset, self)
        return train, test
    
    def get_backbone(self):
        return PTMClassifier(output_size=150, args=self.args)
    
    @staticmethod
    def get_loss():
        return F.cross_entropy


class OnlineCLINC150(ContinualDataset):
    NAME = "online-clinc150"
    SETTING = "instance-il"
    
    NUM_INSTANCE_PER_SLICE = 5
    NUM_TEST_INSTANCE_PER_SLICE = 1
    NUM_VALID_INSTANCE_PER_SLICE = 1
    
    def get_n_tasks(self) -> int:
        n_ps = self.NUM_INSTANCE_PER_SLICE
        n_test_ps = self.NUM_TEST_INSTANCE_PER_SLICE
        n_valid_ps = self.NUM_VALID_INSTANCE_PER_SLICE
        n_train_ps = n_ps - n_test_ps - n_valid_ps
        
        if self.args.filter_rate < 1:
            n_train_ps = int(n_train_ps * self.args.filter_rate)
            n_ps = n_train_ps + n_valid_ps + n_test_ps
            # self.NUM_INSTANCE_PER_SLICE = n_ps
            
        if self.dataset is None:
            self.dataset = CLINC150(mod="train", slice_size=n_ps, ptm=self.args.ptm,
                                    filter_rate=self.args.filter_rate)
            
        slice_num_per_task = max(1, int(self.args.batch_size / n_train_ps))
        n_tasks = int(len(self.dataset.sliced_online_data_idx) / slice_num_per_task)
        return n_tasks
    
    def get_data_loaders(self):
        if self.dataset is None:
            self.dataset = CLINC150(mod="train", slice_size=self.NUM_INSTANCE_PER_SLICE, ptm=self.args.ptm)
    
        train_dataset = CLINC150(quick_load=True)
        test_dataset = CLINC150(quick_load=True)
    
        train, test = store_masked_loaders_for_online(train_dataset, test_dataset, self)
        return train, test
    
    def get_backbone(self):
        return PTMClassifier(output_size=150, args=self.args)
    
    @staticmethod
    def get_loss():
        return F.cross_entropy
