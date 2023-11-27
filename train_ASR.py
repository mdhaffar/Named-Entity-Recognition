#!/usr/bin/env python3
import os
import sys
import torch
import logging
import speechbrain as sb
import torchaudio
import numpy as np
import itertools
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.data_utils import undo_padding
from speechbrain.utils.distributed import run_on_main
from speechbrain.decoders.ctc import filter_ctc_output 

logger = logging.getLogger(__name__)

# Define training procedure
class ASR(sb.core.Brain):
    def compute_forward(self, batch, stage):
        """Forward computations from the waveform batches to the output probabilities."""

        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
   #     wavs, wav_lens = wavs.to(self.device), wav_lens.to(self.device)
        # Forward pass
        self.feats = self.modules.wav2vec2(wavs)
        #,self.hidden_layer1,self.hidden_layer2,self.hidden_layer3,self.hidden_layer4,self.hidden_layer5,self.hidden_layer6,self.hidden_layer7,self.hidden_layer8,self.hidden_layer9,self.hidden_layer10,self.hidden_layer11,self.hidden_layer12,self.hidden_layer13,self.hidden_layer14,self.hidden_layer15,self.hidden_layer16,self.hidden_layer17,self.hidden_layer18,self.hidden_layer19,self.hidden_layer20,self.hidden_layer21,self.hidden_layer22,self.hidden_layer23,self.hidden_layer24,self.hidden_layer25,self.out_CNN = self.modules.wav2vec2(wavs)
 #       self.modules.wav2vec2.requires_grad = True
 #       print("feats:", feats)

        self.y = self.modules.enc(self.feats)
        self.x = self.modules.enc1(self.y)
        logits = self.modules.ctc_lin(self.x)
        p_ctc = self.hparams.log_softmax(logits)
       
        return p_ctc, wav_lens



    def compute_objectives(self, predictions, ids, batch, stage):
        """Computes the CTC loss given predictions and targets."""
        p_ctc, wav_lens = predictions
        chars, char_lens = batch.char_encoded
        
        loss = self.hparams.ctc_cost(
                p_ctc, chars, wav_lens, char_lens
            )
        #self.ctc_metrics.append(batch.id, p_ctc, chars, wav_lens, char_lens)
#        print ("p_ctc: ", p_ctc)
#        print ("wav_lens: ", wav_lens)
        sequence = sb.decoders.ctc_greedy_decode(p_ctc, wav_lens, self.hparams.blank_index)
#        print ("sequence: ", sequence)
#        print ("chars: ", chars)

        if stage != sb.Stage.TRAIN:
#            print ("p_ctc: ", p_ctc)
#            print ("wav_lens: ", wav_lens)

            #sequence = sb.decoders.ctc_greedy_decode(p_ctc, wav_lens, self.hparams.blank_index)
             
#            print ("sequence: ", sequence)
#            print ("chars: ", chars)
            
            '''self.cer_metrics.append(
                ids=batch.id,
                predict=sequence,
                target=chars,
                target_len=char_lens,
                ind2lab=self.label_encoder.decode_ndim,
            )'''
            self.cer_metric.append(
                ids=ids,
                predict=sequence,
                target=chars,
                target_len=char_lens,
                ind2lab=self.label_encoder.decode_ndim
            )
            
            '''print('affichage')
            print(self.label_encoder.decode_ndim)'''
            
            self.coer_metric.append(
                ids=ids,
                predict=sequence,
                target=chars,
                target_len=char_lens,
                ind2lab=self.label_encoder.decode_ndim
            )
            self.cver_metric.append(
                ids=ids,
                predict=sequence,
                target=chars,
                target_len=char_lens,
                ind2lab=self.label_encoder.decode_ndim)
            self.ctc_metric.append(
                ids,
                p_ctc,
                chars,
                wav_lens,
                char_lens
            )

        return loss

    def init_optimizers(self):
        # Initializes the wav2vec2 optimizer and model optimizer.
        self.wav2vec_optimizer = self.hparams.wav2vec_opt_class(
            self.modules.wav2vec2.parameters()
        )
        self.adam_optimizer = self.hparams.adam_opt_class(
            self.hparams.model.parameters()
        )

    def fit_batch(self, batch):
        """Train the parameters given a single batch in input"""
        
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        chars, char_lens = batch.char_encoded
        ids = batch.id

        wavs, wav_lens = wavs.to(self.device), wav_lens.to(self.device)

        stage = sb.Stage.TRAIN

        predictions = self.compute_forward(batch, sb.Stage.TRAIN)
        loss = self.compute_objectives(predictions,ids, batch, sb.Stage.TRAIN)
        loss.backward()
        if self.check_gradients(loss):
            self.wav2vec_optimizer.step()
            self.adam_optimizer.step()

        self.wav2vec_optimizer.zero_grad()
        self.adam_optimizer.zero_grad()

        #YE: loss.detach().cpu() ???
        return loss.detach().cpu()

    def evaluate_batch(self, batch, stage):
        """Computations needed for validation/test batches"""
        # Get data.
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        chars, char_lens = batch.char_encoded
        ids = batch.id

        predictions = self.compute_forward(batch, stage=stage)
        with torch.no_grad():
            loss = self.compute_objectives(predictions,ids, batch, stage=stage)
        return loss.detach()

    def on_stage_start(self, stage, epoch):
        """Gets called when a stage (either training, validation, test) starts."""
        #self.ctc_metrics = self.hparams.ctc_stats()
    
        if stage != sb.Stage.TRAIN:
            self.cer_metric = self.hparams.cer_computer()
            self.ctc_metric = self.hparams.ctc_computer()
            self.coer_metric = self.hparams.coer_computer()
            self.cver_metric = self.hparams.cver_computer()
    
    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of a stage."""
        # Compute/store important stats
        stage_stats = {"loss": stage_loss}

        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss
        else:
            #cer = self.cer_metrics.summarize("error_rate")
            stage_stats["CER"] = self.cer_metric.summarize("error_rate")
            stage_stats["COER"] = self.coer_metric.summarize("error_rate")
            stage_stats["CVER"] = self.cver_metric.summarize("error_rate")

        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:
            '''old_lr_adam, new_lr_adam = self.hparams.lr_annealing_adam(cer)
            old_lr_wav2vec, new_lr_wav2vec = self.hparams.lr_annealing_wav2vec(cer)
            sb.nnet.schedulers.update_learning_rate(self.adam_optimizer, new_lr_adam)
            sb.nnet.schedulers.update_learning_rate(self.wav2vec_optimizer, new_lr_wav2vec)
            '''
            old_lr_adam, new_lr_adam = self.hparams.lr_annealing_adam(stage_stats["loss"])
            old_lr_wav2vec, new_lr_wav2vec = self.hparams.lr_annealing_wav2vec(stage_stats["loss"])
            sb.nnet.schedulers.update_learning_rate(self.adam_optimizer, new_lr_adam)
            sb.nnet.schedulers.update_learning_rate(self.wav2vec_optimizer, new_lr_wav2vec)

            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr_adam": old_lr_adam, "lr_wav2vec": old_lr_wav2vec},
                train_stats={"loss": self.train_loss},
                #valid_stats={"loss": stage_loss, "CER": cer},
                valid_stats=stage_stats,
            )

            self.checkpointer.save_and_keep_only(
                meta={"CER": stage_stats["CER"]}, min_keys=["CER"],
            )
        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )
            '''with open(self.hparams.wer_file, "w") as w:
                w.write("CTC loss stats:\n")
                self.ctc_metrics.write_stats(w)
                w.write("\nCER stats:\n")
                self.cer_metrics.write_stats(w)
                print("CTC and CER stats written to ", self.hparams.wer_file)'''
            with open(hparams["cer_file_test"], "w") as w:
                self.cer_metric.write_stats(w)
            with open(hparams["ctc_file_test"], "w") as w:
                self.ctc_metric.write_stats(w)
            with open(hparams["coer_file_test"], "w") as w:
                self.coer_metric.write_stats(w)
            with open(hparams["cver_file_test"], "w") as w:
                self.cver_metric.write_stats(w)


# Define custom data procedure
def dataio_prepare(hparams):

    # 1. Define datasets
    data_folder = hparams["data_folder"]

    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_csv"], replacements={"data_root": data_folder},
    )

    if hparams["sorting"] == "ascending":
        # we sort training data to speed up training and get better results.
        train_data = train_data.filtered_sorted(
            sort_key="duration",
            key_max_value={"duration": hparams["avoid_if_longer_than"]},
            key_min_value={"duration": hparams["avoid_if_smaller_than"]},
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["dataloader_options"]["shuffle"] = False

    elif hparams["sorting"] == "descending":
        train_data = train_data.filtered_sorted(
            sort_key="duration",
            reverse=True,
            key_max_value={"duration": hparams["avoid_if_longer_than"]},
            key_min_value={"duration": hparams["avoid_if_smaller_than"]},
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["dataloader_options"]["shuffle"] = False

    elif hparams["sorting"] == "random":
        pass

    else:
        raise NotImplementedError(
            "sorting must be random, ascending or descending"
        )

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["valid_csv"], replacements={"data_root": data_folder},
    )
    # We also sort the validation data so it is faster to validate
#    valid_data = valid_data.filtered_sorted(sort_key="duration")
    valid_data = valid_data.filtered_sorted(
        sort_key="duration",
        reverse=True,
        key_max_value={"duration": hparams["avoid_if_longer_than"]},
        key_min_value={"duration": hparams["avoid_if_smaller_than"]},
    )

    test_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["test_csv"], replacements={"data_root": data_folder},
    )
    # We also sort the test data so it is faster to validate
    test_data = test_data.filtered_sorted(
        sort_key="duration",
        reverse=True,
        key_max_value={"duration": hparams["avoid_if_longer_than"]},
        key_min_value={"duration": hparams["avoid_if_smaller_than"]},
    )

    datasets = [train_data, valid_data, test_data]

    label_encoder = sb.dataio.encoder.CTCTextEncoder()

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav", "start_seg", "end_seg")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav, start_seg, end_seg):
#        info = torchaudio.info(wav)
        start = int(float(start_seg) * hparams["sample_rate"])
        stop = int(float(end_seg) * hparams["sample_rate"])
        speech_segment = {"file" : wav, "start" : start, "stop" : stop}
        sig = sb.dataio.dataio.read_audio(speech_segment)
        return sig

#        resampled = torchaudio.transforms.Resample(
#            info.sample_rate, hparams["sample_rate"],
#        )(sig)
#        return resampled

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("char")
    @sb.utils.data_pipeline.provides(
        "char_list", "char_encoded"
    )
    def text_pipeline(char):
        char_list = char.strip().split()
        yield char_list
        char_encoded = label_encoder.encode_sequence_torch(char_list)
        yield char_encoded

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    lab_enc_file = os.path.join(hparams["save_folder"], "label_encoder.txt")
    label_encoder.load_or_create(
        path=lab_enc_file,
        from_didatasets=[train_data],
        output_key="char_list",
        special_labels={"blank_label": hparams["blank_index"]},
        sequence_input=True,
    )

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets, ["id", "sig", "char_encoded"],
    )
    return train_data, valid_data, test_data, label_encoder


if __name__ == "__main__":

    # Load hyperparameters file with command-line overrides
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # If distributed_launch=True then
    # create ddp_group with the right communication protocol
    sb.utils.distributed.ddp_init_group(run_opts)

    # Dataset preparation (parsing CommonVoice)
#    from common_voice_prepare import prepare_common_voice  # noqa

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Due to DDP, we do the preparation ONLY on the main python process
#    run_on_main(
#        prepare_common_voice,
#        kwargs={
#            "data_folder": hparams["data_folder"],
#            "save_folder": hparams["save_folder"],
#            "train_tsv_file": hparams["train_tsv_file"],
#            "dev_tsv_file": hparams["dev_tsv_file"],
#            "test_tsv_file": hparams["test_tsv_file"],
#            "accented_letters": hparams["accented_letters"],
#            "language": hparams["language"],
#        },
#    )

    # Create the datasets objects 
    train_data, valid_data, test_set, label_encoder  = dataio_prepare(hparams)

    # Trainer initialization
    asr_brain = ASR(
        modules=hparams["modules"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # Adding objects to trainer.
    asr_brain.label_encoder = label_encoder
    asr_brain.label_encoder.add_unk()
    # Training
#    with torch.autograd.detect_anomaly():
    asr_brain.fit(
        asr_brain.hparams.epoch_counter,
        train_data,
        valid_data,
        train_loader_kwargs=hparams["dataloader_options"],
        valid_loader_kwargs=hparams["test_dataloader_options"],
    )

    # Test
    asr_brain.hparams.wer_file = hparams["output_folder"] + "/wer_test.txt"
    asr_brain.evaluate(
        test_set,
        min_key="CER",
        test_loader_kwargs=hparams["test_dataloader_options"],
    )
