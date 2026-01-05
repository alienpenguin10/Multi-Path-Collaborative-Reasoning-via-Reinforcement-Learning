
import unittest
from unittest.mock import MagicMock, patch
import torch
from transformers import AutoTokenizer

from trl import GRPOConfig, GRPOTrainer
from tests.testing_utils import TrlTestCase

from transformers import PreTrainedModel, PretrainedConfig

class MockConfig(PretrainedConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = 1000
        self.hidden_size = 16

class MockModel(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.dummy_layer = torch.nn.Linear(1, 1) # Just to have parameters

    def get_input_embeddings(self):
        return MagicMock()

    def get_output_embeddings(self):
        return MagicMock()

    def forward(self, *args, **kwargs):
        return MagicMock(logits=torch.randn(1, 1, 1000))

    def generate(self, *args, **kwargs):
        return torch.tensor([[1]]) # Default return

class TestGRPOHRPO(TrlTestCase, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.model_id = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def test_thinking_embeds_propagation(self):
        # Mocks
        config = MockConfig()
        model = MockModel(config)
        
        # Mock generate to return thinking embeds
        # Signature: prompt_completion_ids, thinking_embeds, thinking_mask, embeds_ratio
        batch_size = 4 # 2 inputs * 2 generations
        seq_len = 10
        hidden_size = 16
        
        prompt_completion_ids = torch.randint(0, 1000, (batch_size, seq_len))
        thinking_embeds = torch.randn(batch_size, seq_len, hidden_size)
        thinking_mask = torch.ones(batch_size, seq_len)
        embeds_ratio = torch.tensor([0.5] * batch_size)
        
        # When unwrap_model_for_generation is called, it returns the model (or wrapped). 
        # We need to ensure the `unwrapped_model.generate` call returns our tuple.
        # GRPOTrainer mocks/wraps model. We can patch `unwrap_model_for_generation` to return our mock.
        
        def generate_side_effect(*args, **kwargs):
            input_ids = kwargs.get("input_ids", args[0] if args else None)
            bs = input_ids.shape[0]
            seq_l = 10
            h_size = 16
            
            p_comp_ids = torch.randint(0, 1000, (bs, seq_l), device=input_ids.device)
            think_emb = torch.randn(bs, seq_l, h_size, device=input_ids.device)
            think_mask = torch.ones(bs, seq_l, device=input_ids.device)
            emb_ratio = torch.tensor([0.5] * bs, device=input_ids.device)
            return p_comp_ids, think_emb, think_mask, emb_ratio

        with patch("trl.trainer.grpo_trainer.unwrap_model_for_generation") as mock_unwrap:
            mock_unwrap.return_value.__enter__.return_value = model
            model.generate = MagicMock(side_effect=generate_side_effect)
            
            # Setup inputs
            dataset = [{"prompt": "Hello"} for _ in range(4)]
            
            
            # Check for GPU availability
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            model = model.to(device)

            training_args = GRPOConfig(
                output_dir=self.tmp_dir,
                learning_rate=0.1,
                per_device_train_batch_size=2,
                num_generations=2,
                max_completion_length=8,
                report_to="none",
                use_vllm=False, 
                use_cpu=False if torch.cuda.is_available() else True
            )
            trainer = GRPOTrainer(
                model=model,
                reward_funcs=lambda **kwargs: [1.0] * len(kwargs["prompts"]),
                train_dataset=dataset,
                args=training_args,
                processing_class=self.tokenizer
            )
            
            # We want to test `_prepare_inputs` or `prediction_step` to see if it handles the thinking embeds
            # But `_prepare_inputs` calls generate.
            
            # Prepare dummy inputs for `_prepare_inputs`
            # It expects a list of dicts per example
            inputs = self.tokenizer(["Hello", "World"], return_tensors="pt", padding=True)
            inputs_list = [
                {
                    "prompt": "Hello", 
                    "prompt_ids": inputs.input_ids[0], 
                    "prompt_mask": inputs.attention_mask[0]
                },
                {
                    "prompt": "World", 
                    "prompt_ids": inputs.input_ids[1], 
                    "prompt_mask": inputs.attention_mask[1]
                }
            ]
            
            # Run _prepare_inputs
            prepared_inputs = trainer._prepare_inputs(inputs_list)
            
            # Verify that thinking_embeds and thinking_mask are in prepared_inputs
            self.assertIn("thinking_embeds", prepared_inputs)
            self.assertIn("thinking_mask", prepared_inputs)
            
            # Verify values
            # prepared_inputs["thinking_embeds"] should be padded list or tensor
            # The existing code converts them to list then pads in compute_loss? 
            # Wait, `_prepare_inputs` returns `output` dict. 
            # In `_prepare_inputs`:
            # output["thinking_embeds"] = pad(thinking_embeds, ...)
            
            self.assertTrue(torch.is_tensor(prepared_inputs["thinking_embeds"]))
            self.assertEqual(prepared_inputs["thinking_embeds"].shape[0], training_args.per_device_train_batch_size) 
            # Wait, GRPOTrainer repeats prompts? 
            # `model.generate` is called with `prompt_ids` (BS*G or just BS?)
            # GRPO: `prompt_ids` passed to `generate` is repeated if `num_generations > 1`.
            # Actually GRPOTrainer repeats input prompts `num_generations` times before creating batches?
            # No, `_prepare_inputs` does:
            # prompt_ids = [p for p in inputs["prompt_ids"] for _ in range(self.num_generations)]
            # So the batch size into `generate` is `2 * 2 = 4`.
            # Our mock returned BS=2. We should adjust mock to return correct shape.
            
            # Let's verify what `_prepare_inputs` does.
            


