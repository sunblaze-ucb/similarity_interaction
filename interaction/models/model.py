import os
os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None) 
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModelForSeq2SeqLM, Gemma3ForCausalLM

torch.use_deterministic_algorithms(False)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("highest") 

class Model:
    def __init__(self, model_id: str, temperature: float = 0):
        if 'Qwen2.5-72B-Instruct' in model_id:
            self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision="d3d951150c1e5848237cd6a7ad11df4836aee842")
        elif 'Phi-4-mini-instruct' in model_id:
            self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision="5a149550068a1eb93398160d8953f5f56c3603e9")
        elif 'Phi-3.5-mini-instruct' in model_id:
            self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision="99361f5cd711db9446bbd790708b50cfdf4365b1")
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        if temperature==0:
            do_sample=False
        else:
            do_sample=True

        self.gen_kwargs = {}
        self.gen_kwargs["do_sample"] = do_sample
        if temperature > 0:
            self.gen_kwargs["temperature"] = temperature

        if self.tokenizer.pad_token is None:    
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.tokenizer.padding_side = "left"
        
        device_map='auto'

        if 'gemma-3' in model_id:
            self.model = Gemma3ForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                device_map=device_map,
                low_cpu_mem_usage=True,
                offload_folder=None, 
                output_hidden_states=False,
                trust_remote_code=True,
                **self.gen_kwargs
            ).eval()
        elif ('microsoft/Phi-3-medium-128k-instruct' in model_id):
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                device_map=device_map,
                low_cpu_mem_usage=True,
                offload_folder=None, 
                output_hidden_states=False,
                trust_remote_code=False,
                attn_implementation="eager"  
            ).eval()
        elif 'Lexius/Phi-3.5-mini-instruct' in model_id:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                revision="99361f5cd711db9446bbd790708b50cfdf4365b1",
                device_map=device_map,
                low_cpu_mem_usage=True,
                offload_folder=None,
                output_hidden_states=False,
                trust_remote_code=False,
                attn_implementation="eager"
            ).eval()
        elif 'Phi-4-mini-instruct' in model_id:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                revision="5a149550068a1eb93398160d8953f5f56c3603e9",
                device_map=device_map,
                low_cpu_mem_usage=True,
                offload_folder=None,
                output_hidden_states=False,
                trust_remote_code=False,
                attn_implementation="eager"
            ).eval()
        elif 'Qwen2.5-72B-Instruct' in model_id:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                revision="d3d951150c1e5848237cd6a7ad11df4836aee842",
                torch_dtype=torch.bfloat16,
                device_map=device_map,
                low_cpu_mem_usage=True,
                offload_folder=None,
                output_hidden_states=False,
                trust_remote_code=True,
            ).eval()
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                device_map=device_map,
                low_cpu_mem_usage=True,
                offload_folder=None, 
                output_hidden_states=False,
                trust_remote_code=True,
            ).eval()

        self.pipe = pipeline(
            "text-generation",
            model=self.model,  
            tokenizer=self.tokenizer,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            **self.gen_kwargs
        )
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.config.eos_token_id = self.tokenizer.eos_token_id

    @torch.inference_mode()
    def get_hidden_states(self, messages):
        #if getattr(self.tokenizer, "chat_template", None):
        #    messages = [[{"role": "user", "content": p}] for p in prompts]
        prompts =  [
            self.tokenizer.apply_chat_template(
                m,
                tokenize=False,
                add_generation_prompt=False,
            )
            for m in messages
        ]
        #else:
        #    prompts2=prompts
        inputs = self.tokenizer(prompts,padding=True, return_tensors="pt").to(self.model.device)
        return inputs, self.model(**inputs, use_cache=False, output_hidden_states=True).hidden_states
    
    @torch.inference_mode()
    def get_hidden_states_last_only(self, messages):
        prompts =  [
            self.tokenizer.apply_chat_template(
                m,
                tokenize=False,
                add_generation_prompt=False,
            )
            for m in messages
        ]
        inputs = self.tokenizer(prompts,padding=True, return_tensors="pt").to(self.model.device)
        last_tok: list[torch.Tensor] = []

        def _grab_last(_, __, output):
            # hidden : (B, T, D) → keep [:, -1, :]
            if isinstance(output, tuple):
                hidden = output[0]          # first item is (B, T, D)
            else:
                hidden = output
            last_tok.append(hidden[:, -1, :].detach().to("cpu"))
        
        hooks = []
        for blk in self.model.model.layers:
            hooks.append(blk.register_forward_hook(_grab_last))
        
        # forward pass through *only* the transformer stack
        _ = self.model.model(
            **inputs,
            use_cache=False
        ) # lm_head is skipped; no logits are produced

        # clean up hooks
        for h in hooks:
            h.remove()
        
        return inputs, torch.stack(last_tok)

    @torch.inference_mode()
    def generate_text(self, messages, max_new_tokens=800):
        generated_texts = []
        #if getattr(self.tokenizer, "chat_template", None):
            #messages = [[{"role": "user", "content": p}] for p in prompts]
        prompts2 =  [
            self.tokenizer.apply_chat_template(
                m,
                tokenize=False,
                add_generation_prompt=True,
            )
            for m in messages
        ]
        #else:
        #    prompts2=messages
        generated = self.pipe(
            prompts2,
            max_new_tokens=max_new_tokens,
            return_full_text=False,
            **self.gen_kwargs
        )
        generated_texts.extend(o[0]["generated_text"] for o in generated)
        return generated_texts
    
    def get_word (self,token):
        return self.tokenizer.decode(token)
    
