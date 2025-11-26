# Qwen3 LLMzip encoder and decoder
# Adapted from llama/LLMzip.py for HuggingFace Transformers

from typing import List
import torch
import numpy as np
import pandas as pd
import zlib
import sys
import binascii
import json

from qwen3.tokenizer import Qwen3Tokenizer
from qwen3.model import Qwen3Model
from qwen3.llmzip_utils import *
from AC.arithmeticcoding import *


class Qwen3_encode:
    def __init__(self, model: Qwen3Model, tokenizer: Qwen3Tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.AC_encoder = None
        self.compression_alg = None
        
    def encode_batch(self, prompt_tokens):
        """
        Encode a batch of token sequences.
        
        Args:
            prompt_tokens: Token sequences, shape (batch_size, seq_len)
            
        Returns:
            ranks: Rank of actual tokens in probability distribution
            probs_tok: Probabilities of actual tokens
        """
        bsz = prompt_tokens.shape[0]   
        prompt_size = prompt_tokens.shape[1]
        
        assert bsz <= self.model.max_batch_size, (bsz, self.model.max_batch_size)
        
        # Convert to tensor and move to device
        device = next(self.model.model.parameters()).device
        tokens = torch.tensor(prompt_tokens, dtype=torch.long, device=device)
        
        cur_pos = prompt_size - 1
        prev_pos = 0
        
        # Get logits for the sequence up to cur_pos
        # We want to predict tokens[cur_pos] given tokens[prev_pos:cur_pos]
        logits = self.model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
        probs = torch.softmax(logits, dim=-1)
        rank = gen_rank(probs, next_token=tokens[:, cur_pos])
        
        probs_np2 = probs.cpu().numpy()
        tokens_np2 = tokens[:, cur_pos].cpu().numpy()
        ranks_np2 = rank.cpu().numpy()
        
        probs_tok = probs_np2[np.arange(bsz), tokens_np2]
        
        # Arithmetic coding
        if (self.compression_alg == 'ArithmeticCoding') or (self.compression_alg == 'both'):
            cumul = np.zeros(self.model.vocab_size + 1, dtype=np.uint64)
            for j in range(bsz):
                prob1 = probs_np2[j]
                cumul[1:] = np.cumsum(prob1 * 10000000 + 1)
                self.AC_encoder.write(cumul, tokens_np2[j])
           
        return ranks_np2, probs_tok
    
    def encode_from_tokens(
        self,
        win_size: int,
        compression_alg: str = 'ArithmeticCoding',
        compressed_file_name: str = 'LLMzip',
        tokens_full=None,
        batched_encode=False,
        with_context_start=False
    ):
        """
        Encode tokens using LLMzip compression.
        
        Args:
            win_size: Context window size
            compression_alg: 'ArithmeticCoding', 'RankZip', or 'both'
            compressed_file_name: Output file name prefix
            tokens_full: Full token sequence to encode
            batched_encode: Whether to use batched encoding
            with_context_start: Whether to exclude initial context from encoding
        """
        self.compression_alg = compression_alg
        self.compressed_file_name = compressed_file_name
        
        win_size_enc = win_size + 1  # Additional 1 for the true token apart from context
        
        if (self.compression_alg == 'ArithmeticCoding') or (self.compression_alg == 'both'):
            self.AC_file_name = compressed_file_name + '_AC.txt'
            file_out = open(self.AC_file_name, 'wb')
            bitout = BitOutputStream(file_out)
            self.AC_encoder = ArithmeticEncoder(32, bitout)
        
        if batched_encode:
            bsz = self.model.max_batch_size
        else:
            bsz = 1
        
        ranks_list = []
        probs_tok_list = []
        
        n_runs = tokens_full.size - win_size_enc + 1
        
        if not with_context_start:
            tokens_encoded = tokens_full
            
            # Running LLM for the starter tokens
            for t_ind in range(1, win_size_enc):
                tokens_in = np.array([[self.tokenizer.bos_id] + tokens_full[:t_ind].tolist()])
                ranks, probs_tok = self.encode_batch(tokens_in)
                ranks_list += [ranks]
                probs_tok_list += [probs_tok]
            starter_tokens = None
        else:
            tokens_encoded = tokens_full[win_size:win_size + n_runs]
            starter_tokens = tokens_full[:win_size]
        
        n_batches = np.ceil(n_runs / bsz).astype(int)
        
        for b_ind in range(n_batches):
            batch_range_start = b_ind * bsz
            batch_range_stop = np.minimum(n_runs, (b_ind + 1) * bsz)
            tokens_batch = np.array([
                tokens_full[i:i + win_size_enc] 
                for i in range(batch_range_start, batch_range_stop)
            ])
            ranks, probs_tok = self.encode_batch(tokens_batch)
            ranks_list += [ranks]
            probs_tok_list += [probs_tok]
            
            if (b_ind * bsz * 100 / n_batches) % 10 == 0:
                print(f'Encoder: Completed {int(b_ind * bsz * 100 / n_batches)} %')
        
        ranks_full = np.concatenate(ranks_list, 0).squeeze()
        probs_tok_full = np.concatenate(probs_tok_list, 0).squeeze()
        
        if (self.compression_alg == 'ArithmeticCoding') or (self.compression_alg == 'both'):
            self.AC_encoder.finish()
            bitout.close()
            file_out.close()
        
        if (self.compression_alg == 'RankZip') or (self.compression_alg == 'both'):
            str_ranks = get_str_array(ranks_full)
            rank_bytes = bytes(str_ranks, 'ascii')
            ranks_comp = zlib.compress(rank_bytes, 9)
            
            self.RZ_file_name = compressed_file_name + '_RZ.txt'
            
            with open(self.RZ_file_name, 'wb') as file_out_zip:
                file_out_zip.write(ranks_comp)
        
        self.compute_compression_ratio(tokens_encoded, probs_tok_full, starter_tokens)
    
    def compute_compression_ratio(self, tokens_encoded, probs_tok, starter_tokens):
        """Compute and save compression metrics"""
        text_encoded = self.tokenizer.decode(tokens_encoded.squeeze().tolist())
        
        N_T = tokens_encoded.size
        N_C = len(text_encoded)
        
        df_out = {}
        df_out['$N_C$'] = [N_C]
        df_out['$N_T$'] = [N_T]
        df_out['$H_{ub}$'] = [str(np.sum(-1 * np.log2(probs_tok)) / N_C)]
        
        if (self.compression_alg == 'RankZip') or (self.compression_alg == 'both'):
            with open(self.RZ_file_name, 'rb') as file_RZ:
                ranks_compressed_bytes = file_RZ.read()
            rho_RZ = len(ranks_compressed_bytes) * 8 / N_C
            print(f'Compression Ratio for RankZip: {rho_RZ} bits/char')
            
            df_out['Qwen3+zlib compressed file size'] = [len(ranks_compressed_bytes) * 8]
            df_out['$\rho_{Qwen3+Zlib}$'] = [rho_RZ]
        
        df_out['$\rho_{TbyT}$'] = [str(np.sum(np.ceil(-1 * np.log2(probs_tok))) / N_C)]
        
        if (self.compression_alg == 'ArithmeticCoding') or (self.compression_alg == 'both'):
            b_ind = 1
            file_in = open(self.AC_file_name, 'rb')
            bitin = BitInputStream(file_in)
            compressed_bits = read_bitstream(bitin)
            rho_AC = compressed_bits.size / N_C
            print(f'Compression Ratio for Arithmetic Coding: {rho_AC} bits/char')
            file_in.close()
            
            df_out['Qwen3+AC compressed file size'] = [compressed_bits.size]
            df_out['$\rho_{Qwen3+AC}$'] = [rho_AC]
        
        print(df_out)
        
        with open(self.compressed_file_name + '_metrics.json', 'w') as file_metrics:
            json.dump(df_out, file_metrics)


class Qwen3_decode:
    def __init__(self, model: Qwen3Model, tokenizer: Qwen3Tokenizer):
        self.model = model
        self.tokenizer = tokenizer
    
    def decode_AC(
        self,
        win_size,
        starter_tokens,
        total_length: int,
        compressed_file_name: str = 'LLMzip_AC.txt'
    ):
        """
        Decode from arithmetic coding compressed file.
        
        Args:
            win_size: Context window size
            starter_tokens: Initial context tokens (or None)
            total_length: Total number of tokens to decode
            compressed_file_name: Input compressed file name
            
        Returns:
            Decoded text string
        """
        file_in = open(compressed_file_name, 'rb')
        bitin = BitInputStream(file_in)
        dec = ArithmeticDecoder(32, bitin)
        
        bsz = 1  # Predicts 1 token at a time
        
        if starter_tokens is not None:
            total_length += win_size
        
        device = next(self.model.model.parameters()).device
        tokens = torch.full((bsz, total_length), self.tokenizer.pad_id, dtype=torch.long, device=device)
        bos_token = torch.full((bsz, 1), self.tokenizer.bos_id, dtype=torch.long, device=device)
        
        cumul = np.zeros(self.model.vocab_size + 1, dtype=np.uint64)
        probs_list = []
        
        if starter_tokens is None:
            start_pos = 0
            prev_pos = -1
        else:
            tokens[:, :win_size] = torch.tensor(starter_tokens, dtype=torch.long, device=device)
            start_pos = win_size
            prev_pos = 0
        
        for cur_pos in range(start_pos, total_length):
            if prev_pos == -1:
                logits = self.model.forward(bos_token, 0)
                prev_pos += 1
            elif cur_pos < win_size:
                logits = self.model.forward(
                    torch.cat((bos_token, tokens[:, prev_pos:cur_pos]), 1), 0
                )
            else:
                logits = self.model.forward(tokens[:, prev_pos:cur_pos], 0)
            
            probs = torch.softmax(logits, dim=-1)
            probs_np = probs.cpu().numpy().reshape((-1,))
            
            probs_list += [probs_np]
            cumul[1:] = np.cumsum(probs_np * 10000000 + 1)
            next_token = dec.read(cumul, probs_np.size)
            
            tokens[:, cur_pos] = torch.tensor(next_token, dtype=torch.long, device=device)
            if cur_pos >= win_size:
                prev_pos += 1
                
                if (prev_pos * 100 / (total_length - win_size)) % 10 == 0:
                    print(f'Decoder: Completed {int(prev_pos * 100 / (total_length - win_size))} %')
        
        decoded_text = self.tokenizer.decode(tokens.tolist()[0])
        
        bitin.close()
        file_in.close()
        
        return decoded_text
    
    def decode_ranks(
        self,
        win_size,
        starter_tokens,
        compressed_file_name: str = 'LLMzip_RZ.txt'
    ):
        """
        Decode from RankZip compressed file.
        
        Args:
            win_size: Context window size
            starter_tokens: Initial context tokens (or None)
            compressed_file_name: Input compressed file name
            
        Returns:
            Decoded text string
        """
        with open(compressed_file_name, 'rb') as file_in:
            ranks_compressed = file_in.read()
        
        ranks_decomp = zlib.decompress(ranks_compressed).decode('ascii')
        ranks_in = np.fromstring(ranks_decomp, sep=' ', dtype=np.int64)
        
        bsz = 1  # Predicts 1 token at a time
        
        total_length = ranks_in.shape[0]
        
        device = next(self.model.model.parameters()).device
        
        if starter_tokens is None:
            ranks = torch.tensor(ranks_in, device=device).reshape(bsz, -1)
        else:
            total_length += win_size
            ranks_in = np.append(np.zeros((win_size,), dtype=np.int64), ranks_in)
            ranks = torch.tensor(ranks_in, device=device).reshape(bsz, -1)
        
        bos_token = torch.full((bsz, 1), self.tokenizer.bos_id, dtype=torch.long, device=device)
        tokens = torch.full((bsz, total_length), self.tokenizer.pad_id, dtype=torch.long, device=device)
        
        if starter_tokens is None:
            start_pos = 0
            prev_pos = -1
        else:
            tokens[:, :win_size] = torch.tensor(starter_tokens, dtype=torch.long, device=device)
            start_pos = win_size
            prev_pos = 0
        
        for cur_pos in range(start_pos, total_length):
            if prev_pos == -1:
                logits = self.model.forward(bos_token, 0)
                prev_pos += 1
            elif cur_pos < win_size:
                logits = self.model.forward(
                    torch.cat((bos_token, tokens[:, prev_pos:cur_pos]), 1), 0
                )
            else:
                logits = self.model.forward(tokens[:, prev_pos:cur_pos], 0)
            
            probs = torch.softmax(logits, dim=-1)
            next_token = gen_next_token(probs, ranks[:, cur_pos:cur_pos + 1])
            tokens[:, cur_pos] = torch.tensor(next_token, dtype=torch.long, device=device)
            
            if cur_pos >= win_size:
                prev_pos += 1
                
                if (prev_pos * 100 / (total_length - win_size)) % 10 == 0:
                    print(f'Decoder: Completed {int(prev_pos * 100 / (total_length - win_size))} %')
        
        decoded_text = self.tokenizer.decode(tokens.tolist()[0])
        
        return decoded_text
