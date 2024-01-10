from dataclasses import dataclass, field

import numpy as np
import torch
import transformers
from transformers import GenerationConfig
import pickle

from train import ModelArguments, smart_tokenizer_and_embedding_resize, DEFAULT_PAD_TOKEN, DEFAULT_EOS_TOKEN, \
  DEFAULT_BOS_TOKEN, DEFAULT_UNK_TOKEN, PROMPT_DICT


@dataclass
class InferenceArguments:
  model_max_length: int = field(
    default=512,
    metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
  )
  load_in_8bit: bool = field(
    default=False,
    metadata={"help": "Load the model in 8-bit mode."},
  )
  inference_dtype: torch.dtype = field(
    default=torch.float32,
    metadata={"help": "The dtype to use for inference."},
  )

def generate_statistics(out, input, gen, tokenizer):

    l0 = [min(torch.argwhere(input[i, :] == 29901)) for i in range(input.shape[0])] # To get tokens of first integer. 29901 decodes to ":".
    l1 = [torch.argwhere(input[i, :] == 718) for i in range(input.shape[0])] # To get tokens of integer after +, i.e. second integer. 718 decodes to "+".
    l2 = [torch.argwhere(input[i, ] == 353) for i in range(input.shape[0])] # Final token before addition sum starts. 353 decodes to "=".

    input_text = [tokenizer.decode(input[i, l0[i]+2:l2[i]+1]) for i in range(input.shape[0])]

    correctness = [sum([int(s) for s in input_text[j].split() if s.isdigit()]) == [int(s) for s in gen[j].replace('</s>', ' ').replace(',', '').split() if s.isdigit()][-1] for j in range(len(input_text))]
    acc = sum(correctness) / len(input_text)

    add_att_ex = []
    add_att_max_ex = []

    ### We pick out the staircase part of the attention patterns based on whether the relevant entry is non-zero or not.

    for i in range(len(out)): 
        add_att = []
        add_att_max = []
        for j in range(len(out[i])):
            out_all_att = out[i][j].squeeze()

            if j == 0:

                for k in range(4):
                    s = torch.argwhere(out_all_att[:, :, l1[i].item()+2+k, l0[i].item()+2+k] > 0)
                    sm = out_all_att[s[:, 0], s[:, 1], l1[i].item()+2+k, l0[i].item()+2+k]
                    add_att.append(s[torch.argmax(sm), :])
                    add_att_max.append(torch.max(sm))

                # for k in range(4):
                s = torch.argwhere(out_all_att[:, :, -1, l1[i].item()+2+j] > 0)
                sm = out_all_att[s[:, 0], s[:, 1], -1, l1[i].item()+2+j]
                add_att.append(s[torch.argmax(sm), :])
                add_att_max.append(torch.max(sm))

            else:

                # for k in range(4):
                s = torch.argwhere(out_all_att[:, :, l1[i].item()+2+j] > 0)
                sm = out_all_att[s[:, 0], s[:, 1], l1[i].item()+2+j]
                add_att.append(s[torch.argmax(sm), :])
                add_att_max.append(torch.max(sm))
        
        add_att_ex.append(add_att)
        add_att_max_ex.append(add_att_max) 

    counter = torch.zeros((32, 32))
    for i in range(len(out)):
        for j in range(len(out[i])+4):
            counter[add_att_ex[i][j][0].item(), add_att_ex[i][j][1].item()] += 1

    return input_text, correctness, l0, l1, l2, acc, counter, add_att_ex, add_att_max_ex

def generate_prompt(instruction, input=None):
  if input:
    return PROMPT_DICT["prompt_input"].format(instruction=instruction, input=input)
  else:
    return PROMPT_DICT["prompt_no_input"].format(instruction=instruction)


def inference():
  parser = transformers.HfArgumentParser((ModelArguments, InferenceArguments))
  model_args, inference_args = parser.parse_args_into_dataclasses()

  model = transformers.AutoModelForCausalLM.from_pretrained(
    model_args.model_name_or_path,
    load_in_8bit=inference_args.load_in_8bit,
    torch_dtype=inference_args.inference_dtype,
    device_map="auto",
  )
  model.cuda()
  model.eval()

  tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_args.model_name_or_path,
    use_fast=False,
    model_max_length=inference_args.model_max_length,
  )

  if tokenizer.pad_token is None:
    smart_tokenizer_and_embedding_resize(
      special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
      tokenizer=tokenizer,
      model=model,
    )
  tokenizer.add_special_tokens(
    {
      "eos_token": DEFAULT_EOS_TOKEN,
      "bos_token": DEFAULT_BOS_TOKEN,
      "unk_token": DEFAULT_UNK_TOKEN,
    }
  )
  
  generator = torch.Generator().manual_seed(42)
  ints = torch.randint(1000, 10000, size=(1000, 2), generator=generator)  
  
  prompts = ["{!s} + {!s} = ".format(a0.item(), a1.item()) for a0, a1 in zip(ints[:, 0], ints[:, 1])]

  ctx = ""
  i = 0
  Responses = []
  input_ = torch.tensor([])
  gen_ = []
  gen_tokens = []
  att = []
  for instruction in prompts:
    att_z = []
    print("Instruction:", instruction)
    inputs = tokenizer(generate_prompt(instruction, None), return_tensors="pt")
    generation = model.generate(input_ids=inputs["input_ids"].cuda(),
                            max_new_tokens=inference_args.model_max_length,
                            return_dict_in_generate=True,
                            output_attentions=True)
    
    output = generation.attentions
    
    for i in range(len(output)):
      att_p = torch.tensor([])
      for j in range(len(output[0])):
          att_p = torch.cat((att_p, output[i][j].clone().detach().to('cpu').unsqueeze(0)), 0)
      att_z.append(att_p.unsqueeze(0))
    
    att.append(att_z)
  
    input_ = torch.cat((input_, inputs['input_ids']), 0)

    input_length = 1 if model.config.is_encoder_decoder else inputs.input_ids.shape[1]
    generated_tokens = generation.sequences[:, input_length:]
    gen = tokenizer.decode(generated_tokens[0])
    
    gen_.append(gen)
    gen_tokens.append(generated_tokens[0])

    ctx += f"Instruction: {instruction}\n" + f"Response: {generated_tokens[0]}\n"
    print("Response:", gen)
    print()
    Responses.append(gen)

  input_text, correctness, l0, _, l2, acc, counter, _, _ = generate_statistics(att, input_, gen_, tokenizer=tokenizer)
  
  example = 0
  seq_len = input_[example].shape[0]

  sorted_ = torch.sort(counter.view(-1), descending=True)

  ll0 = l0[example].item() + 2
  ll2 = l2[example].item() 

  av_att = []
  std_att = []
  for i in range(2):
      average_attention = torch.zeros(seq_len - ll0 + 5, seq_len - ll0 + 5)
      std_attention = torch.zeros(seq_len - ll0 + 5, seq_len - ll0 + 5)

      s = 0
      for j in range(5):
          coord = torch.argwhere(counter == sorted_[0][i])[0]
          adder = torch.cat([att[i][j].squeeze()[coord[0], coord[1]].unsqueeze(0) for i in range(len(prompts))], 0)
          if j == 0:
              adder = adder[:, ll0:seq_len, ll0:seq_len]
              _, a, b = adder.shape
          else:
              a = 1
              adder = adder[:, ll0:seq_len+j]
              _, b = adder.shape
          average_attention[s:s+a, :b] = adder.mean(0)
          std_attention[s:s+a, :b] = adder.std(0)
          s += a
      av_att.append(average_attention)
      std_att.append(std_attention)

  Data = [input_text, correctness, acc, counter, av_att, std_att]

  path = "" ### PUT YOUR OWN PATH HERE ###
  torch.save(Data, path + "data.pt")
  torch.save(input_, path + "inputs.pt")
  torch.save(gen_, path + "generations.pt")
  torch.save(gen_tokens, path + "gen_tokens.pt")

if __name__ == "__main__":
  inference()
