use candle_core::{Device, Tensor};
use candle_transformers::generation::LogitsProcessor;
pub use candle_transformers::models::quantized_t5::{
    Config, T5EncoderModel, T5ForConditionalGeneration, VarBuilder,
};
use tokenizers::Tokenizer;

pub struct ModelConditionalGeneration {
    model: T5ForConditionalGeneration,
    tokenizer: Tokenizer,
    config: Config,
}

impl ModelConditionalGeneration {
    pub fn load(
        weights: Vec<u8>,
        tokenizer: Vec<u8>,
        config: Vec<u8>,
    ) -> Result<ModelConditionalGeneration, String> {
        let device = &Device::Cpu;
        let vb = match VarBuilder::from_gguf_buffer(&weights, device) {
            Ok(vb) => vb,
            Err(e) => {
                return Err(format!("Failed to create var builder: {:?}", e.to_string()));
            }

        };
        let mut config: Config = match serde_json::from_slice(&config) {
            Ok(config) => config,
            Err(e) => {
                return Err(format!("Failed to parse config: {:?}", e.to_string()));
            }
        };
        let tokenizer =
            Tokenizer::from_bytes(&tokenizer).map_err(|m| m.to_string())?;
        let model = match T5ForConditionalGeneration::load(vb, &config) {
            Ok(model) => model,
            Err(e) => {
                return Err(format!("Failed to load model: {:?}", e.to_string()));
            }

        };
        config.use_cache = false;
        Ok(Self {
            model,
            tokenizer,
            config,
        })
    }
    pub fn decode(&mut self, input: ConditionalGenerationParams) -> Result<ConditionalGenerationOutput, String> {
        let device = &Device::Cpu;
        self.model.clear_kv_cache();
        let mut output_token_ids = [self.config.pad_token_id as u32].to_vec();
        let prompt = input.prompt;
        let repeat_penalty = input.repeat_penalty;
        let repeat_last_n = input.repeat_last_n;
        let seed = input.seed;
        let max_length = usize::clamp(input.max_length.unwrap_or(512), 0, 512);
        let temperature = if input.temperature <= 0. {
            None
        } else {
            Some(input.temperature)
        };
        let top_p = if input.top_p <= 0. || input.top_p >= 1. {
            None
        } else {
            Some(input.top_p)
        };
        let mut logits_processor = LogitsProcessor::new(seed, temperature, top_p);
        let tokens = self
            .tokenizer
            .encode(prompt, true)
            .map_err(|m| m.to_string())?
            .get_ids()
            .to_vec();

        let tsr = match Tensor::new(&tokens[..], device) {
            Ok(tsr) => tsr,
            Err(e) => {
                return Err(format!("Failed to create tensor: {:?}", e.to_string()));
            }
        };
        let input_token_ids = match tsr.unsqueeze(0) {
            Ok(tsr) => tsr,
            Err(e) => {
                return Err(format!("Failed to unsqueeze tensor: {:?}", e.to_string()));
            }

        };
        let encoder_output = match self.model.encode(&input_token_ids) {
            Ok(tsr) => tsr,
            Err(e) => {
                return Err(format!("Failed to encode tensor: {:?}", e.to_string()));
            }

        };
        let mut decoded = String::new();
        for index in 0.. {
            if output_token_ids.len() > max_length {
                break;
            }
            let decoder_token_ids = if index == 0 {
                match match Tensor::new(output_token_ids.as_slice(), device) {
                    Ok(tsr) => tsr,
                    Err(e) => {
                        return Err(format!("Failed to create tensor: {:?}", e.to_string()));
                    }

                }.unsqueeze(0) {
                    Ok(tsr) => tsr,
                    Err(e) => {
                        return Err(format!("Failed to unsqueeze tensor: {:?}", e.to_string()));
                    }
                }
            } else {
                let last_token = *output_token_ids.last().unwrap();
                match match Tensor::new(&[last_token], device){
                    Ok(tsr) => tsr,
                    Err(e) => {
                        return Err(format!("Failed to create tensor: {:?}", e.to_string()));
                    }
                }.unsqueeze(0) {
                    Ok(tsr) => tsr,
                    Err(e) => {
                        return Err(format!("Failed to unsqueeze tensor: {:?}", e.to_string()));
                    }
                }
            };
            let logits = match self
                .model
                .decode(&decoder_token_ids, &encoder_output) {
                    Ok(tsr) => tsr,
                    Err(e) => {
                        return Err(format!("Failed to decode tensor: {:?}", e.to_string()));
                    }
                }.squeeze(0);
            let logits = if repeat_penalty == 1. {
                logits
            } else {
                let start_at = output_token_ids.len().saturating_sub(repeat_last_n);
                candle_transformers::utils::apply_repeat_penalty(
                    &logits.unwrap(),
                    repeat_penalty,
                    &output_token_ids[start_at..],
                )
            };

            let next_token_id = match logits_processor.sample(&logits.unwrap()) {
                Ok(tsr) => tsr,
                Err(e) => {
                    return Err(format!("Failed to sample tensor: {:?}", e.to_string()));
                }
            };
            if next_token_id as usize == self.config.eos_token_id {
                break;
            }
            output_token_ids.push(next_token_id);
            if let Some(text) = self.tokenizer.id_to_token(next_token_id) {
                let text = text.replace('▁', " ").replace("<0x0A>", "\n");
                decoded += &text;
            }
        }
        Ok(
            ConditionalGenerationOutput {
                generation: decoded,
            },
        )
    }
}

pub struct ConditionalGenerationOutput {
    pub generation: String,
}

pub struct ConditionalGenerationParams {
    pub prompt: String,
    pub temperature: f64,
    pub seed: u64,
    pub top_p: f64,
    pub repeat_penalty: f32,
    pub repeat_last_n: usize,
    pub max_length: Option<usize>,
}