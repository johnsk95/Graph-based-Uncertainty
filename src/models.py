from abc import ABC, abstractmethod
from openai import OpenAI
import torch
import os
import src.utils as utils
from dotenv import load_dotenv

load_dotenv()

class BaseModel(ABC):
    def __init__(self, model_name, args):
        self.model_name = model_name
        self.args = args

    @abstractmethod
    def generate_given_prompt(self, prompt):
        pass

    @abstractmethod
    def generate_n_given_prompt(self, prompt):
        pass

class OpenAIModel(BaseModel):
    def __init__(self, model_name, args):
        super().__init__(model_name, args)
        self.client = OpenAI(
            # This is the default and can be omitted
            api_key=os.environ.get("OPENAI_API_KEY"),
        )
        if model_name == 'gpt-3.5-turbo':
            self.model_name = 'gpt-3.5-turbo-0125'

    def generate_given_prompt(self, prompt):
        # print('Prompt\n', prompt)
        if type(prompt) == str:
            prompt = [{'role': 'user', 'content': prompt}]
            
        if 'gpt' in self.model_name:
            response = self.client.chat.completions.create(
                messages=prompt,
                model=self.model_name,
                max_tokens=1000,
                temperature=0.0,
            )
        else:
            raise NotImplementedError
        
        return {'generation': response.choices[0].message.content, 'prompt': prompt}

    def generate_n_given_prompt(self, prompt):
        if 'gpt' in self.model_name:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{'role': 'user', 'content': prompt}],
                max_tokens=1000,
                temperature=self.args.temperature,
                n=self.args.num_generations_per_prompt
            )
        else:
            raise NotImplementedError
   
        return {'generation': [choice.message.content for choice in response.choices], 'prompt': prompt}


class Llama3Model(BaseModel):
    def __init__(self, model_name, args):
        super().__init__(model_name, args)
        if 'llama-3-70b-instruct' == model_name:
            self.model, self.tokenizer = utils.load_llama3_70b_model_and_tokenizer()
        elif 'llama-3-8b-instruct' == model_name:
            self.model, self.tokenizer = utils.load_llama3_8b_model_and_tokenizer()

    def generate_given_prompt(self, prompt):
        messages = [{"role": "user", "content": prompt}]
        input_ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)
        
        torch.cuda.empty_cache()

        terminators = [
            self.tokenizer.eos_token_id,
        ]

        outputs = self.model.generate(
            input_ids, max_new_tokens=1000, eos_token_id=terminators, 
            do_sample=True, temperature=0.0001, pad_token_id=self.tokenizer.eos_token_id
        )
        generation = self.tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True).strip()
        
        return {'generation': generation, 'prompt': prompt}

    def generate_n_given_prompt(self, prompt):
        messages = [{"role": "user", "content": prompt}]
        input_ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)

        terminators = [
            self.tokenizer.eos_token_id,
        ]

        outputs = self.model.generate(
            input_ids, max_new_tokens=1000, eos_token_id=terminators, 
            do_sample=True, temperature=self.args.temperature, 
            num_return_sequences=self.args.num_generations_per_prompt, 
            pad_token_id=self.tokenizer.eos_token_id
        )
        generations = [self.tokenizer.decode(decoded[input_ids.shape[-1]:], skip_special_tokens=True).strip() for decoded in outputs]
        return {'generation': generations, 'prompt': prompt}

def get_model(model_name, args):
    if 'gpt' in model_name:
        return OpenAIModel(model_name, args)
    elif 'llama-3' in model_name:
        return Llama3Model(model_name, args)
    else:
        raise NotImplementedError

if __name__ == '__main__':
    model_name = 'gpt-3.5-turbo'
    args = None
    model = get_model(model_name, args)
    prompt = 'What is the meaning of life?'
    results = model.generate_given_prompt(prompt)
    print(results)
    more_results = model.generate_n_given_prompt(prompt)
    print(more_results)
