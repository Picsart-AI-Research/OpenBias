from llama import Llama

class Llama_2:
    def __init__(
        self,
        ckpt_dir,
        tokenizer_path,
        max_seq_len,
        max_batch_size,
        model_parallel_size,
        rank,
        max_gen_len,
        temperature,
        top_p,
        seed,
        SYSTEM_PROMPT
    ) -> None:
        
        self.generator = Llama.build(
            ckpt_dir=ckpt_dir,
            tokenizer_path=tokenizer_path,
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            model_parallel_size=model_parallel_size,
            local_rank=rank,
            seed=seed,
        )
        self.max_gen_len = max_gen_len
        self.temperature = temperature
        self.top_p = top_p
        self.SYSTEM_PROMPT = SYSTEM_PROMPT
    
    def generate(self, sentences):
        dialogs = []
        for sentence in sentences:
            sentence = sentence.replace("\n","")
            dialogs.append(
                self.SYSTEM_PROMPT + [
                    {
                        'role': 'user',
                        'content': f'prompt: "{sentence}". Remember to answer in JSON format only providing multiple classes.'
                    }
                ]
            )
        return self.generator.chat_completion(
            dialogs,  # type: ignore
            max_gen_len=self.max_gen_len,
            temperature=self.temperature,
            top_p=self.top_p,
        )

    def update_system_prompt(self, system_prompt):
        self.SYSTEM_PROMPT = system_prompt