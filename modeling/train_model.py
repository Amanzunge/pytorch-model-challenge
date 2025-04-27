#train_model.py
import torch

from data.events_loader import EventsLoader

torch.set_float32_matmul_precision('high')
torch.set_default_dtype( torch.bfloat16 )

import time
from datetime import datetime
import os

from modeling.models.model import ModelTrainer

from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter

base_path = os.path.dirname(os.path.abspath(__file__))

def simple_collate(batch): #Windows multiprocessing cannot pickle lambda functions
    tgts, srcs_list, masks_list = zip(*batch)

    batched_srcs = {}
    for key in srcs_list[0].keys():
        batched_srcs[key] = torch.stack([src[key] for src in srcs_list], dim=0)

    batched_tgts = torch.stack(tgts, dim=0)

    return batched_tgts, batched_srcs, None


def load_config(config_path = f'{base_path}/training_config.json' ):
    with open(config_path, 'r') as f:
        import json
        config = json.load(f)
    return config


def main():
    config = load_config()
    print( config )
    full_dataset = EventsLoader(
        batch_size = config['batch_size'],
        src_seq_length = config[ 'src_seq_len' ],
        tgt_seq_length = config[ 'tgt_seq_len' ],
        id_category_size = config['id_category_size']
    )
    print( full_dataset.category_fields.values() )
    output_names = full_dataset.category_fields.keys()
    model_g = ModelTrainer(
        d_input = full_dataset.input_size,
        d_categories = list( full_dataset.category_fields.values() ),
        d_output = full_dataset.output_size,
        d_model = config['model']['d_model'],
        n_heads = config['model']['num_heads'],
        encoder_layers = config['model']['encoder_layers'],
        decoder_layers = config['model']['decoder_layers'],
        encoders = full_dataset.encoder_streams
    )

    start_epoch = 0
    if config['model_checkpoint'] is not None:
        model_loaded = torch.load(
            config['model_checkpoint'],
            weights_only = False,
            map_location = torch.device('cpu')
        )
        model_g.load_state_dict(
            model_loaded[ 'model_state_dict' ],
        )
        start_epoch = model_loaded['epoch']

    model_g = model_g.to( 'cuda', dtype = torch.bfloat16 )
    optimizer_g = torch.optim.Adam(
        model_g.parameters(),
        lr = 3e-4
    )
    print(f"Model device: {next(model_g.parameters()).device}") #CHECKING CUDA
    model_g.train()
    #full_dataset.load_dataset() #dont need this because of synthetic data
    training_data, test_data = torch.utils.data.random_split( full_dataset, [ 0.9, 0.1 ] )
    train_dataloader = DataLoader(
        full_dataset,
        num_workers = 8, #Been 24
        batch_size = 8,
        shuffle = False,
        persistent_workers = True,
        pin_memory = True,
        collate_fn = simple_collate #changed from lambda x: x[0]
    )
    test_dataloader = DataLoader(
        test_data,
        num_workers = 8, 
        batch_size = 8,
        shuffle = False,
        persistent_workers = True,
        pin_memory = True,
        collate_fn = simple_collate #changed from lambda x: x[0]
    )

    current_datetime = datetime.now()
    run_directory = f'runs/{current_datetime.strftime("%Y-%m-%d")}_{current_datetime.strftime("%H-%M")}' #Changed this also Windows folder creation
    test_writer = SummaryWriter( log_dir = f'{run_directory}/test' )
    train_writer = SummaryWriter( log_dir = f'{run_directory}/train' )
    train(
        model_g,
        optimizer_g,
        train_writer,
        test_writer,
        train_dataloader,
        test_dataloader,
        start_epoch,
        config['epochs'],
        config['grad_accum'],
        run_directory,
        output_names,
        config['batch_size']
    )


def train(
    model_g,
    optimizer_g,
    train_writer,
    test_writer,
    train_dataloader,
    test_dataloader,
    start_epoch,
    epochs,
    grad_accum,
    run_directory,
    output_names,
    batch_size
):
    # Model Initialization
    for epoch in range( start_epoch, epochs ):
        print(f'epoch {epoch}')
        execute(
            grad_accum,
            epoch,
            model_g,
            test_dataloader,
            test_writer,
            output_names,
            batch_size
        )
        optimizer_g.zero_grad()
        execute(
            grad_accum,
            epoch,
            model_g,
            train_dataloader,
            train_writer,
            output_names,
            batch_size,
            optimizer_g, True
        )

        test_writer.flush()
        train_writer.flush()

        try:
            if epoch % 10 == 0:
                torch.save( {
                    'epoch': epoch,
                    'model_state_dict': model_g.state_dict(),
                    'optimizer_state_dict': optimizer_g.state_dict(),
                }, f'{run_directory}/forcaster_checkpoint_{epoch}.pt' )

            torch.save( {
                'epoch': epoch,
                'model_state_dict': model_g.state_dict(),
                'optimizer_state_dict': optimizer_g.state_dict(),
            }, f'{run_directory}/forcaster_checkpoint_latest.pt' )
        except Exception as e:
            print( e )
            pass


def execute(
    accum_iter,
    epoch,
    model_g: "ModelTrainer",
    dataloader,
    writer,
    output_names,
    batch_size,
    optimizer_g = None,
    train = False,
):
    last_time = time.time()
    total_steps = len( dataloader )
    print( total_steps )
    for data_step, (
        tgt,
        srcs,
        masks
    ) in enumerate( dataloader ):
        full_progress = count_steps( epoch, data_step, total_steps, batch_size )
        writer.add_scalar( f"Time/Data", (time.time() - last_time), full_progress )
        last_time = time.time()

        ''' Train Forcaster '''
        if optimizer_g is not None:
            (
                category_loss,
                category_loss_separated,
             ) = model_g( srcs, tgt, masks, train ) #Added []
        else:
            with torch.no_grad():
                (
                    category_loss,
                    category_loss_separated
                ) = model_g( srcs, tgt, masks ) #Added []
        print(f"[{'Train' if train else 'Test'}] Epoch {epoch}, Step {data_step}/{total_steps}, Loss: {category_loss.item():.4f}") #Added this
        writer.add_scalar( f"Model/category_loss", category_loss, full_progress )
        for i, name in enumerate( output_names ):
            writer.add_scalar( f"Model/{name}", category_loss_separated[i], full_progress )

        if optimizer_g is not None:
            if ((data_step + 1) % accum_iter == 0) or (data_step + 1 == total_steps):
                optimizer_g.step()
                optimizer_g.zero_grad()

        writer.add_scalar( f"Time/Compute", (time.time() - last_time), full_progress )
        last_time = time.time()


def count_steps( epoch: int, batch_num, num_batches, batch_size ):
    return (epoch * num_batches + batch_num) * batch_size


if __name__ == '__main__':
    main()
    
