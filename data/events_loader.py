#events_loader.py
from torch.utils.data import Dataset
import torch

class EventsLoader(Dataset):

    def __init__(
        self,
        batch_size,
        src_seq_length,
        tgt_seq_length,
        id_category_size
    ):
        self.batch_size = batch_size
        self.src_sequence_length = src_seq_length
        self.tgt_sequence_length = tgt_seq_length
        self.id_category_size = id_category_size

        self.encoder_streams = [
            'EventType',
            'RecipientRecordId',
            'LocationRecordId',
            'CostCodeRecordId',
            'JobCodeRecordId',
        ]

        self.datetime_fields = [
            'EventDatetime',
            'ReferenceDatetime'
        ]
        self.employee_ids_columns = [
            'ActorRecordId',
            'RecipientRecordId',
            'ApprovedByManagerUserRecordId'
        ]

        self.initial_category_fields = {
            'EventType': 10,
            'ActorRecordId': int(id_category_size),
            'RecipientRecordId': int(id_category_size),
            'LocationRecordId': int(id_category_size),
            'CostCodeRecordId': int(id_category_size),
            'JobCodeRecordId': int(id_category_size),
            'ApprovedByManagerUserRecordId': int(id_category_size),
            'IsTimesheet': 4,
            'HoursWorked': 100,
        }

        self.category_fields = {
            **self.initial_category_fields,
            'Time_Event_Month': 14,
            'Time_Event_Day': 33,
            'Time_Event_Hour': 25,
            'Time_Event_Minute': 61,
            'Time_Reference_Month': 14,
            'Time_Reference_Day': 33,
            'Time_Reference_Hour': 25,
            'Time_Reference_Minute': 61
        }

        self.input_size = 1
        self.output_size = max(self.category_fields.values())

    def __len__(self):
        return 500

    def __getitem__(self, idx):
        src_streams = {}

        for stream in self.encoder_streams:
            src = torch.randint(
                low=0,
                high=self.id_category_size,
                size=(self.src_sequence_length, self.input_size),
                dtype=torch.long
            )
            src_streams[stream] = src

        masks = None

        # Generate targets based on src_streams
        target_classes = []

        for i, field in enumerate(self.category_fields.keys()):
            if field == 'EventType':
                # Pick first event token from EventType stream
                target = src_streams['EventType'][0, 0] % self.category_fields[field]
            elif field == 'ActorRecordId':
                # Use random token from ActorRecordId stream
                target = src_streams['RecipientRecordId'][1, 0] % self.category_fields[field]
            elif field == 'LocationRecordId':
                target = src_streams['LocationRecordId'][2, 0] % self.category_fields[field]
            elif field == 'CostCodeRecordId':
                target = (src_streams['CostCodeRecordId'][3, 0] + src_streams['JobCodeRecordId'][4, 0]) % self.category_fields[field]
            else:
                # For remaining fields, just random for now
                target = torch.randint(0, size=(1,), high=self.category_fields[field]).squeeze(0)
        
            target_classes.append(target)

        target_classes = torch.stack(target_classes)

        return (
            target_classes,  # shape [17]
            src_streams,
            masks
        )