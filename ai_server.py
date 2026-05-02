# =====================
# ai_server.py - AI SERVER FOR BROWSER EXTENSION
# =====================

from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
import torch
import logging
from datetime import datetime
import os
import threading
import time
import re
import json
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Optional

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
# Configure CORS to allow browser extension
CORS(app, resources={
    r"/api/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    },
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent.absolute()

class AIChatbot:
    def __init__(self, model_path=None, training_data_path=None):
        # Auto-detect paths relative to script location
        if model_path is None:
            # Point to parent directory, not checkpoint - code will find checkpoint automatically
            model_path = str(SCRIPT_DIR / "models" / "distilbert-misinfo")
        if training_data_path is None:
            training_data_path = str(SCRIPT_DIR / "training_data_processed.json")
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.model_type = "unknown"
        self.classifier_model = None
        self.classifier_tokenizer = None
        self.classifier_model_id = os.environ.get('CLASSIFIER_MODEL_ID', 'dhruvpal/fake-news-bert')
        self.fact_check_api_key = os.environ.get('FACT_CHECK_API_KEY')
        self.factcheck_debug = str(os.environ.get('FACTCHECK_DEBUG', '')).strip().lower() in {'1', 'true', 'yes', 'y'}
        self.training_data_path = training_data_path
        self.misinformation_db = {}  # Load from training data
        self._load_misinformation_db()
        self.load_model()
        self.load_classifier()
        # Best-effort warmup in background to reduce first-call latency
        try:
            threading.Thread(target=self._warmup, daemon=True).start()
        except Exception:
            pass

    def load_classifier(self):
        try:
            self.classifier_tokenizer = AutoTokenizer.from_pretrained(self.classifier_model_id)
            self.classifier_model = AutoModelForSequenceClassification.from_pretrained(self.classifier_model_id)
            self.classifier_model.eval()
            logger.info(f"✅ Classifier model '{self.classifier_model_id}' loaded successfully!")
        except Exception as e:
            logger.warning(f"⚠️ Could not load classifier model '{self.classifier_model_id}': {e}")
            self.classifier_model = None
            self.classifier_tokenizer = None

    def classify_text(self, text):
        if not self.classifier_model or not self.classifier_tokenizer:
            return None

        inputs = self.classifier_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )

        with torch.no_grad():
            outputs = self.classifier_model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]

        pred_idx = int(torch.argmax(probs).item())
        confidence = float(probs[pred_idx].item())

        # Determine which index corresponds to FAKE/REAL using id2label when available
        misinfo_idx = None
        factual_idx = None
        try:
            id2label = getattr(self.classifier_model.config, 'id2label', None)
            if isinstance(id2label, dict) and id2label:
                for idx, label in id2label.items():
                    try:
                        i = int(idx)
                    except Exception:
                        continue
                    l = str(label).strip().lower()
                    if any(k in l for k in ['fake', 'false', 'misinfo', 'misinformation', 'rumor', 'rumour']):
                        misinfo_idx = i
                    if any(k in l for k in ['real', 'true', 'factual', 'fact', 'legit', 'legitimate']):
                        factual_idx = i
        except Exception:
            pass

        # Fallback to common convention if mapping not present
        if misinfo_idx is None and factual_idx is None:
            misinfo_idx = 1
            factual_idx = 0

        is_misinformation = (pred_idx == misinfo_idx)

        # Report confidence for the predicted class; also keep explicit class probabilities if needed later
        confidence = float(probs[pred_idx].item())
        return {
            'is_misinformation': is_misinformation,
            'confidence': confidence
        }

    def fetch_fact_checks(self, query, max_results=3):
        """Fetch fact-check results using Google Fact Check Tools API.

        Requires FACT_CHECK_API_KEY environment variable.
        Returns list of dicts with publisher, url, title, rating.
        """
        if not self.fact_check_api_key:
            return []

        q = (query or '').strip()
        if not q:
            return []

        try:
            params = {
                'query': q,
                'pageSize': str(max_results),
                'key': self.fact_check_api_key
            }
            url = 'https://factchecktools.googleapis.com/v1alpha1/claims:search?' + urllib.parse.urlencode(params)
            req = urllib.request.Request(url, headers={'User-Agent': 'MisinformationChatbot/1.0'})
            with urllib.request.urlopen(req, timeout=8) as resp:
                body = resp.read().decode('utf-8', errors='replace')
            data = json.loads(body)
        except Exception as e:
            logger.warning(f"Fact Check API error: {e}")
            return []

        results = []
        for claim in (data.get('claims') or []):
            for review in (claim.get('claimReview') or []):
                results.append({
                    'title': (review.get('title') or '').strip(),
                    'url': (review.get('url') or '').strip(),
                    'publisher': ((review.get('publisher') or {}).get('name') or '').strip(),
                    'rating': (review.get('textualRating') or '').strip(),
                    'claim': (claim.get('text') or '').strip()
                })
                if len(results) >= max_results:
                    break
            if len(results) >= max_results:
                break

        return results

    def infer_verdict_from_factchecks(self, fact_checks):
        """Infer verdict from fact-check textual ratings.
        Returns: True for misinformation, False for factual, None for unknown.
        """
        if not fact_checks:
            return None

        false_hits = 0
        true_hits = 0
        for fc in fact_checks:
            r = (fc.get('rating') or '').lower()
            if any(k in r for k in ['false', 'pants on fire', 'incorrect', 'fake', 'misleading', 'scam', 'hoax', 'wrong', 'not true']):
                false_hits += 1
            if any(k in r for k in ['true', 'correct', 'accurate', 'mostly true']):
                true_hits += 1

        if false_hits > true_hits and false_hits > 0:
            return True
        if true_hits > false_hits and true_hits > 0:
            return False
        return None

    def build_factcheck_queries(self, message):
        """Build a small set of compact queries for the Fact Check Tools API.

        The API tends to work best with short claim-like queries.
        """
        text = (message or '').strip()
        if not text:
            return []

        # First sentence / clause
        first = re.split(r'(?<=[.!?])\s+|\n+', text, maxsplit=1)[0].strip()
        first = re.sub(r'\s+', ' ', first)
        first = first[:180].strip(' .,:;\t')

        # Keyword-based fallback
        words = re.findall(r"[A-Za-z0-9']+", text)
        stop = {
            'the','a','an','and','or','but','if','then','else','when','while','for','to','of','in','on','at','by','with',
            'as','is','are','was','were','be','been','being','it','this','that','these','those','they','them','their',
            'he','she','his','her','you','we','i','me','my','our','your','from','into','about','over','under','during',
            'saw','see','seen','said','says','claim','claims','rumor','rumors','viral','social','media'
        }
        keywords = []
        for w in words:
            wl = w.lower()
            if wl in stop:
                continue
            if len(wl) < 4:
                continue
            keywords.append(w)
            if len(keywords) >= 10:
                break

        kw_query = ' '.join(keywords).strip()
        kw_query = kw_query[:180].strip()

        queries = []
        if first and len(first) >= 8:
            queries.append(first)
        if kw_query and kw_query.lower() != first.lower():
            queries.append(kw_query)
        return queries
    
    def _load_misinformation_db(self):
        """Load both misinformation and factual information from training dataset (risk-based)"""
        try:
            if os.path.exists(self.training_data_path):
                with open(self.training_data_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Extract unique claims with risk-based responses (both misinformation and factual)
                for item in data:
                    # Extract the core claim from question (remove prefixes)
                    question = item['question']
                    # Remove question prefixes
                    claim = re.sub(r'^(is this accurate|verify this claim|fact-check this|is this information reliable|analyze the accuracy of|what is true about):\s*', 
                                  '', question, flags=re.IGNORECASE).strip()
                    
                    is_misinformation = item.get('is_misinformation', False)
                    
                    # Store risk-based responses
                    if is_misinformation:
                        # Misinformation responses
                        responses = {
                            'low': item.get('low_risk_response', ''),
                            'medium': item.get('medium_risk_response', ''),
                            'high': item.get('high_risk_response', '')
                        }
                    else:
                        # Factual information responses
                        responses = {
                            'low': item.get('low_risk_response', ''),
                            'medium': item.get('medium_risk_response', ''),
                            'high': item.get('high_risk_response', '')
                        }
                    
                    # Ensure all response types have content
                    for key in ['low', 'medium', 'high']:
                        if not responses[key]:
                            # Fallback to expert_response if available
                            if item.get('expert_response'):
                                responses[key] = item.get('expert_response')
                            elif item.get('medium_risk_response'):
                                responses[key] = item.get('medium_risk_response')
                            else:
                                responses[key] = "Information not available."
                    
                    # Store by normalized claim text with type indicator
                    claim_key = claim.lower()
                    full_key = f"{'MISINFO' if is_misinformation else 'FACTUAL'}:{claim_key}"
                    if full_key not in self.misinformation_db:
                        self.misinformation_db[full_key] = {
                            'responses': responses,
                            'is_misinformation': is_misinformation,
                            'claim': claim
                        }
                
                # Count types
                misinfo_count = sum(1 for v in self.misinformation_db.values() if v['is_misinformation'])
                factual_count = len(self.misinformation_db) - misinfo_count
                logger.info(f"✅ Loaded {len(self.misinformation_db)} claims from training data: {misinfo_count} misinformation, {factual_count} factual")
            else:
                logger.warning(f"Training data file not found: {self.training_data_path}")
        except Exception as e:
            logger.warning(f"Could not load information database: {e}")
            self.misinformation_db = {}
    
    def _check_known_misinformation(self, message, risk_level=None):
        """Check if message matches known information from training data (both misinformation and factual)
        Returns tuple: (response, is_personalized) where is_personalized indicates if response was already personalized
        """
        # Normalize incoming message similarly to how claims are stored
        # (remove common prompt prefixes like "Verify this claim:")
        cleaned_message = re.sub(
            r'^(is this accurate|verify this claim|fact-check this|is this information reliable|analyze the accuracy of|what is true about):\s*',
            '',
            message,
            flags=re.IGNORECASE
        ).strip()
        msg_lower = cleaned_message.lower()
        
        # Direct match first
        for full_key, claim_data in self.misinformation_db.items():
            claim = claim_data['claim'].lower()
            responses = claim_data['responses']
            is_misinformation = claim_data['is_misinformation']
            
            # Check if the claim is mentioned in the message
            # Extract key terms from claim (remove common words)
            claim_words = set(re.findall(r'\b\w+\b', claim))
            msg_words = set(re.findall(r'\b\w+\b', msg_lower))
            common_words = {'this', 'is', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'was', 'were', 'are', 'can', 'that', 'these', 'those'}
            
            claim_keywords = claim_words - common_words
            msg_keywords = msg_words - common_words
            
            # Fast path: direct substring match
            if claim and claim in msg_lower:
                overlap = 1.0
            elif len(claim_keywords) > 0:
                # Check if at least 60% of claim keywords are in message
                overlap = len(claim_keywords & msg_keywords) / len(claim_keywords)
            else:
                overlap = 0.0

            if overlap >= 0.6:  # 60% keyword match
                # Get appropriate response based on risk level
                if risk_level:
                    risk_normalized = risk_level.lower()
                    # Select response based on risk level
                    if risk_normalized == 'high':
                        response = responses.get('high', responses.get('medium', responses.get('low', '')))
                    elif risk_normalized == 'medium':
                        response = responses.get('medium', responses.get('low', ''))
                    elif risk_normalized == 'low':
                        response = responses.get('low', '')
                    else:
                        response = responses.get('medium', '')
                    
                    info_type = "misinformation" if is_misinformation else "factual information"
                    logger.info(f"📋 Known {info_type} found! Using {risk_level} risk response")
                    return (response, True)  # Response is already risk-appropriate, no need for additional personalization
                else:
                    # No risk level provided - use medium risk response as default
                    response = responses.get('medium', responses.get('low', ''))
                    info_type = "misinformation" if is_misinformation else "factual information"
                    logger.info(f"📋 Known {info_type} found but no risk_level provided - returning medium risk response")
                    return (response, False)
        
        return (None, False)
    
    def _clean_response(self, answer):
        """Clean up repetitive or low-quality responses"""
        if not answer or len(answer.strip()) < 10:
            return None
        
        # Remove number sequences (1 2 3 4...)
        answer = re.sub(r'\b\d+\s+\d+\s+\d+\s+\d+.*', '', answer).strip()
        
        # Remove duplicate sentences
        sentences = [s.strip() for s in answer.split('.') if s.strip()]
        unique_sentences = []
        seen = set()
        
        for sentence in sentences:
            if re.match(r'^\d+(\s+\d+)*$', sentence):  # Skip number-only sentences
                continue
            sentence_lower = sentence.lower()
            if sentence_lower not in seen:
                unique_sentences.append(sentence)
                seen.add(sentence_lower)
        
        if not unique_sentences:
            return None
        
        return '. '.join(unique_sentences) + '.'
    
    def _is_low_quality(self, answer, message):
        """Check if response is low quality or unrelated"""
        if not answer or len(answer) < 20:
            return True
        
        # Check for unrelated content indicators
        unrelated = ['citi', 'dea', 'mueller', 'comey', 'fbi director', 'http://', 'www.']
        if any(ind in answer.lower() for ind in unrelated):
            return True
        
        # Check keyword overlap with question
        msg_words = set(re.findall(r'\b\w+\b', message.lower()))
        ans_words = set(re.findall(r'\b\w+\b', answer.lower()))
        common = {'this', 'is', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        msg_keywords = msg_words - common
        ans_keywords = ans_words - common
        
        if len(msg_keywords) > 0:
            overlap = len(msg_keywords & ans_keywords) / len(msg_keywords)
            if overlap < 0.2:  # Less than 20% overlap = likely unrelated
                return True
        
        return False

    def _warmup(self):
        try:
            sample = "User: hello\nAssistant:"
            inputs = self.tokenizer(sample, return_tensors="pt", max_length=32, truncation=True)
            with torch.no_grad():
                _ = self.model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                    max_new_tokens=1,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
        except Exception:
            pass

    def _load_from_huggingface(self, repo_id: str, subfolder: Optional[str] = None) -> bool:
        """Load causal LM from Hugging Face Hub (set MISINFO_HF_REPO, optional MISINFO_HF_SUBFOLDER)."""
        try:
            extra = f" (subfolder={subfolder})" if subfolder else ""
            logger.info(f"Loading AI model from Hugging Face Hub: {repo_id}{extra}")
            kwargs = {}
            if subfolder:
                kwargs["subfolder"] = subfolder
            self.tokenizer = AutoTokenizer.from_pretrained(repo_id, **kwargs)
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            self.model = AutoModelForCausalLM.from_pretrained(
                repo_id,
                torch_dtype=dtype,
                **kwargs,
            )
            self.model.eval()
            if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model_type = "trained"
            logger.info("✅ AI model loaded from Hugging Face Hub successfully!")
            return True
        except Exception as e:
            logger.warning(f"Hugging Face Hub model load failed: {e}")
            return False
    
    def load_model(self):
        """Load your trained AI model"""
        # Allow forcing a small public fallback model to ensure fast startup
        force_fallback = str(os.getenv("USE_FALLBACK_MODEL", "")).lower() in ("1", "true", "yes", "on")
        
        if force_fallback:
            logger.info("Using fallback model (forced by environment variable)")
            self._load_fallback_model()
            return

        hf_repo = (os.environ.get("MISINFO_HF_REPO") or "").strip()
        hf_sub = (os.environ.get("MISINFO_HF_SUBFOLDER") or "").strip() or None
        if hf_repo:
            if self._load_from_huggingface(hf_repo, hf_sub):
                return
            logger.warning("Hub load failed; trying local model path next.")

        # Try to find the model - check for checkpoint directories
        model_to_load = None
        
        if os.path.isdir(self.model_path):
            # Check if there are checkpoint directories
            try:
                items = os.listdir(self.model_path)
                checkpoints = [d for d in items 
                              if os.path.isdir(os.path.join(self.model_path, d)) and d.startswith('checkpoint-')]
                
                if checkpoints:
                    # Use the latest checkpoint (highest number)
                    checkpoint_nums = []
                    for c in checkpoints:
                        try:
                            num = int(c.split('-')[1])
                            checkpoint_nums.append((num, c))
                        except (ValueError, IndexError):
                            continue
                    
                    if checkpoint_nums:
                        # Sort by number and get the latest
                        checkpoint_nums.sort(key=lambda x: x[0], reverse=True)
                        latest_checkpoint = checkpoint_nums[0][1]
                        model_to_load = os.path.join(self.model_path, latest_checkpoint)
                        logger.info(f"✅ Found checkpoint directory: {latest_checkpoint} (step {checkpoint_nums[0][0]})")
                    else:
                        # Use the first checkpoint if we can't parse numbers
                        model_to_load = os.path.join(self.model_path, checkpoints[0])
                        logger.info(f"Using checkpoint directory: {checkpoints[0]}")
                else:
                    # Check if model files exist in root directory
                    required_files = ['config.json', 'tokenizer.json']
                    has_files = all(os.path.exists(os.path.join(self.model_path, f)) for f in required_files)
                    if has_files:
                        model_to_load = self.model_path
                        logger.info(f"Loading model from root directory: {self.model_path}")
                    else:
                        # Check if we're pointing to a checkpoint directory - look in parent
                        parent_dir = os.path.dirname(self.model_path)
                        if parent_dir and os.path.exists(parent_dir):
                            # Check parent for config/tokenizer, use current path for model weights
                            parent_has_files = all(os.path.exists(os.path.join(parent_dir, f)) for f in required_files)
                            if parent_has_files or os.path.exists(os.path.join(self.model_path, "model.safetensors")):
                                # Try loading from checkpoint with parent config
                                model_to_load = self.model_path
                                logger.info(f"Loading model from checkpoint: {self.model_path} (config from parent)")
                            else:
                                logger.warning(f"Model directory exists but no checkpoints or model files found in {self.model_path}")
                        else:
                            logger.warning(f"Model directory exists but no checkpoints or model files found in {self.model_path}")
            except Exception as e:
                logger.warning(f"Error checking model directory: {e}")
        
        # Try loading the local model
        if model_to_load and os.path.exists(model_to_load):
            try:
                logger.info(f"Loading AI model from: {model_to_load}")
                # Check if this is a checkpoint (has model.safetensors but no config.json)
                is_checkpoint = os.path.exists(os.path.join(model_to_load, "model.safetensors")) and not os.path.exists(os.path.join(model_to_load, "config.json"))
                
                if is_checkpoint:
                    # For checkpoints, we need to load config/tokenizer from base model or allow download
                    # Try to load tokenizer - allow download if not found locally
                    try:
                        self.tokenizer = AutoTokenizer.from_pretrained(model_to_load, local_files_only=True)
                    except Exception:
                        # If not found locally, try to infer base model or allow download
                        # For DistilBERT models, try common base models
                        base_models = ["distilbert-base-uncased", "distilgpt2"]
                        tokenizer_loaded = False
                        for base_model in base_models:
                            try:
                                logger.info(f"Trying to load tokenizer from base model: {base_model}")
                                self.tokenizer = AutoTokenizer.from_pretrained(base_model)
                                tokenizer_loaded = True
                                break
                            except Exception:
                                continue
                        if not tokenizer_loaded:
                            # Last resort: try loading from checkpoint without local_files_only
                            logger.warning("Loading tokenizer from checkpoint (may download config files)")
                            self.tokenizer = AutoTokenizer.from_pretrained(model_to_load, local_files_only=False)
                else:
                    # Full model directory - load normally
                    self.tokenizer = AutoTokenizer.from_pretrained(model_to_load, local_files_only=True)
                
                # Load model - for checkpoints, allow loading weights even if config not local
                if is_checkpoint:
                    try:
                        self.model = AutoModelForCausalLM.from_pretrained(
                            model_to_load, 
                            local_files_only=True,
                            use_safetensors=True,
                            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                        )
                    except Exception as e:
                        # If local load fails, try without local_files_only (will download config)
                        logger.warning(f"Local model load failed, trying with config download: {e}")
                        self.model = AutoModelForCausalLM.from_pretrained(
                            model_to_load, 
                            local_files_only=False,
                            use_safetensors=True,
                            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                        )
                else:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_to_load, 
                        local_files_only=True,
                        use_safetensors=True,
                        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                    )
                self.model_type = "trained"
                logger.info("✅ AI model loaded from local path successfully!")
                logger.info(f"   Model type: Trained custom model")
                
                # Ensure pad token exists to avoid generation errors
                if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                
                return
            except Exception as e:
                logger.warning(f"Failed to load local model: {e}")
                logger.info("Falling back to public model...")
        
        # Fallback to a small public model
        self._load_fallback_model()
    
    def _load_fallback_model(self):
        """Load fallback public model"""
        fallback_model = "distilgpt2"
        try:
            logger.warning(f"Loading fallback model: {fallback_model}")
            logger.warning(f"⚠️  NOTE: Fallback model has limited capabilities and may produce repetitive output.")
            logger.warning(f"   For best results, ensure your trained model is in: {self.model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(fallback_model)
            self.model = AutoModelForCausalLM.from_pretrained(fallback_model)
            self.model_type = "fallback"
            
            # Ensure pad token exists
            if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            logger.info(f"✅ Fallback model '{fallback_model}' loaded successfully!")
            logger.info(f"   Model type: Basic fallback (limited functionality)")
        except Exception as e:
            logger.error(f"❌ Failed to load even fallback model: {e}")
            raise RuntimeError(f"Could not load any model. Original error: {e}")
    
    def generate_response(self, message, risk_level=None, susceptibility_score=None):
        """Generate AI response personalized based on user's susceptibility score (risk level)"""
        try:
            # If risk_level isn't provided, derive it from susceptibility_score (0-100)
            if (not risk_level) and (susceptibility_score is not None):
                try:
                    score = float(susceptibility_score)
                    if score < 30:
                        risk_level = 'low'
                    elif score < 70:
                        risk_level = 'medium'
                    else:
                        risk_level = 'high'
                except (TypeError, ValueError):
                    pass

            if risk_level:
                risk_level = str(risk_level).lower()
                if risk_level not in {'low', 'medium', 'high'}:
                    risk_level = None

            classification = self.classify_text(message)
            if classification is not None:
                is_misinformation = classification['is_misinformation']
                confidence = classification['confidence']

                # Evidence layer (Google Fact Check Tools)
                fact_checks = []
                evidence_note = None
                if self.fact_check_api_key:
                    tried_queries = self.build_factcheck_queries(message)
                    if not tried_queries:
                        tried_queries = [message[:180]]

                    used_query = None
                    for q in tried_queries:
                        fact_checks = self.fetch_fact_checks(q, max_results=3)
                        if fact_checks:
                            used_query = q
                            break

                    inferred = self.infer_verdict_from_factchecks(fact_checks)
                    # If fact-checkers strongly disagree, trust them over the classifier
                    if inferred is not None and inferred != is_misinformation:
                        logger.info("Fact-check evidence overrides classifier verdict")
                        is_misinformation = inferred
                        # Keep classifier confidence but don't present it as absolute
                        confidence = min(confidence, 0.99)
                    if not fact_checks:
                        evidence_note = "No matching fact-check results were found for this text."
                        if tried_queries and self.factcheck_debug:
                            logger.info("Fact-check query miss. Tried queries: %s", " | ".join([q[:120] for q in tried_queries]))
                    else:
                        if used_query:
                            if self.factcheck_debug:
                                logger.info("Fact-check results found for query: %s", used_query)
                            evidence_note = "Fact-check results were found for this claim."
                else:
                    evidence_note = "Fact Check API is not configured (set FACT_CHECK_API_KEY)."

                # Avoid presenting classifier probabilities as absolute truth
                confidence = min(float(confidence), 0.99)

                if risk_level is None:
                    risk_level = 'medium'

                if is_misinformation:
                    if risk_level == 'low':
                        answer = f"❌ MISINFORMATION (confidence {confidence:.0%}). This claim is likely false."
                    elif risk_level == 'medium':
                        answer = f"❌ MISINFORMATION (confidence {confidence:.0%}). This claim is likely false. Please verify using reliable sources (reputable news outlets, official institutions, peer-reviewed research, trusted fact-checkers)."
                    else:
                        answer = f"❌ MISINFORMATION (confidence {confidence:.0%}). This claim is likely false and could be harmful if acted on. Verify with trusted sources (official institutions, academic journals, established fact-checkers) and be cautious about sharing it further."
                else:
                    if risk_level == 'low':
                        answer = f"✅ FACTUAL (confidence {confidence:.0%}). This claim is likely true."
                    elif risk_level == 'medium':
                        answer = f"✅ FACTUAL (confidence {confidence:.0%}). This claim is likely true. If you want to confirm, cross-check with reputable sources."
                    else:
                        answer = f"✅ FACTUAL (confidence {confidence:.0%}). This claim is likely true. For high-stakes decisions, confirm using primary sources and reputable institutions, and compare multiple trusted references."

                if fact_checks:
                    # Add evidence section (more evidence for higher risk/detail)
                    max_cites = 1 if risk_level == 'low' else (2 if risk_level == 'medium' else 3)
                    cite_lines = []
                    for fc in fact_checks[:max_cites]:
                        pub = fc.get('publisher') or 'Fact-check'
                        rating = fc.get('rating') or 'Rating unavailable'
                        url = fc.get('url')
                        title = fc.get('title')
                        label = f"{pub} — {rating}"
                        if title:
                            label = f"{label}: {title}"
                        if url:
                            cite_lines.append(f"- {label} ({url})")
                        else:
                            cite_lines.append(f"- {label}")

                    if cite_lines:
                        answer = answer + "\n\nEvidence:\n" + "\n".join(cite_lines)
                elif evidence_note and risk_level in {'medium', 'high'}:
                    answer = answer + "\n\nEvidence:\n- " + evidence_note

                return {
                    "success": True,
                    "response": answer,
                    "is_misinformation": is_misinformation,
                    "is_factual": (not is_misinformation),
                    "confidence": confidence,
                    "fact_checks": fact_checks,
                    "evidence_note": evidence_note,
                    "risk_level": risk_level,
                    "personalized": risk_level is not None,
                    "timestamp": datetime.now().isoformat()
                }

            # Check for known misinformation first (fast path)
            # Returns tuple: (response, is_personalized)
            known_response, already_personalized = self._check_known_misinformation(message, risk_level)
            
            # Generate model response with risk level context
            if risk_level:
                risk_context = {
                    'low': 'Provide a minimal response: just state if it is misinformation or factual.',
                    'medium': 'Provide a brief explanation with basic context.',
                    'high': 'Provide a detailed explanation with warnings and educational content.'
                }
                risk_instruction = risk_context.get(risk_level.lower(), '')
                prompt = f"User: {message}\nContext: {risk_instruction}\nAssistant:"
            else:
                prompt = f"User: {message}\nAssistant:"
            inputs = self.tokenizer(prompt, return_tensors="pt", max_length=256, truncation=True, padding=True)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                    max_new_tokens=150,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.85,
                    top_k=50,
                    repetition_penalty=1.5,
                    no_repeat_ngram_size=4,
                    pad_token_id=self.tokenizer.eos_token_id if self.tokenizer.pad_token_id is None else self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            
            # Decode response
            input_length = inputs["input_ids"].shape[1]
            generated_tokens = outputs[0][input_length:]
            answer = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
            
            # Clean response
            answer = self._clean_response(answer)
            
            # If we found a known misinformation response (especially if already personalized), use it
            # This ensures personalized responses are always used for known misinformation
            if known_response:
                logger.info(f"✅ Using known misinformation response (personalized: {already_personalized})")
                answer = known_response
            # Quality check - if low quality and no known response, use fallback
            elif not answer or self._is_low_quality(answer, message):
                answer = "I'm having difficulty analyzing this claim. Please try rephrasing or consult reliable fact-checking sources."
            
            # Detect misinformation from response content
            answer_lower = answer.lower()
            misinfo_keywords = ['false', 'misinformation', 'not true', 'debunked', 'untrue', 'incorrect', 
                              'inaccurate', 'hoax', 'myth', 'conspiracy', 'unfounded', 'baseless']
            factual_keywords = ['true', 'factual', 'accurate', 'verified', 'confirmed', 'reliable', 
                              'supported', 'evidence', 'proven', 'correct', 'valid']
            
            # If we used a known response, check if it's misinformation or factual
            if known_response:
                # Check the response content to determine if it's misinformation or factual
                is_misinformation = any(kw in answer_lower for kw in misinfo_keywords)
                if is_misinformation:
                    logger.info("✅ Known misinformation detected - marking as misinformation")
                else:
                    logger.info("✅ Known factual information detected - marking as factual")
                    is_misinformation = False  # Explicitly set to False for factual content
            else:
                # Otherwise, detect from response content
                is_misinformation = any(kw in answer_lower for kw in misinfo_keywords)
                # If no misinformation keywords found, check if it's factual
                if not is_misinformation:
                    is_factual_detected = any(kw in answer_lower for kw in factual_keywords)
                    if is_factual_detected:
                        is_misinformation = False
                        logger.info("✅ Factual information detected - marking as factual")
            
            is_factual = any(kw in answer_lower for kw in factual_keywords)
            
            # Personalize response based on risk level (only if not already personalized)
            if is_misinformation and risk_level and not already_personalized:
                logger.info(f"🔄 Personalizing response for risk level: {risk_level} (not yet personalized)")
                answer = self._personalize_response(answer, message, risk_level, susceptibility_score)
            elif already_personalized:
                logger.info(f"✅ Response already personalized for risk level: {risk_level}")
            elif is_misinformation and not risk_level:
                logger.warning("⚠️ Misinformation detected but no risk_level provided - using base response")
            elif not is_misinformation:
                logger.info("ℹ️ No misinformation detected - no personalization needed")
            
            return {
                "success": True,
                "response": answer,
                "is_misinformation": is_misinformation,
                "is_factual": is_factual,
                "risk_level": risk_level,
                "personalized": risk_level is not None,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                "success": False,
                "error": str(e),
                "response": "Sorry, I encountered an error processing your request."
            }
    
    def _personalize_response(self, base_response, message, risk_level, susceptibility_score):
        """Personalize response based on user's susceptibility score"""
        # Normalize risk level (handle both 'High' and 'high')
        risk_level_normalized = risk_level.lower() if risk_level else 'medium'
        logger.info(f"🔍 Personalizing response - Original Risk: {risk_level}, Normalized: {risk_level_normalized}, Score: {susceptibility_score}")
        
        # Extract the core claim from message
        claim = message
        # Remove question prefixes
        claim = re.sub(r'^(is this accurate|verify this claim|fact-check this|is this information reliable|analyze the accuracy of|fact-check|verify|check):\s*', 
                      '', claim, flags=re.IGNORECASE).strip()
        
        if risk_level_normalized == 'high':
            logger.info("✅ Applying HIGH RISK personalization - detailed warnings and educational content")
            # High risk: More detailed, educational, with warnings and sources
            # Start with the base response (which already says it's false/misinformation)
            enhanced_response = base_response
            
            # Add comprehensive warning and educational content
            enhanced_response += "\n\n⚠️ **Important Warning**: Believing such claims can be harmful to your health and well-being. "
            enhanced_response += "There is no scientific evidence supporting this claim. "
            enhanced_response += "Trusted medical and scientific organizations (like WHO, CDC, and peer-reviewed research) confirm that "
            enhanced_response += "this type of information is false and potentially dangerous.\n\n"
            
            enhanced_response += "**Why this matters**: Following unverified health claims can lead to:\n"
            enhanced_response += "• Delayed proper medical treatment\n"
            enhanced_response += "• Worsening of health conditions\n"
            enhanced_response += "• Financial loss from ineffective treatments\n"
            enhanced_response += "• Spread of misinformation to others\n\n"
            
            enhanced_response += "**What you should do**:\n"
            enhanced_response += "• Consult qualified healthcare professionals for medical advice\n"
            enhanced_response += "• Verify information with reputable sources (WHO, CDC, medical journals)\n"
            enhanced_response += "• Be skeptical of claims that seem too good to be true\n"
            enhanced_response += "• Question sources that lack scientific backing\n\n"
            
            enhanced_response += "Would you like to see verified sources or learn more about how to identify reliable health information?"
            
            return enhanced_response
            
        elif risk_level_normalized == 'medium':
            logger.info("✅ Applying MEDIUM RISK personalization - moderate warnings")
            # Medium risk: Moderate detail with some warnings
            enhanced_response = base_response
            enhanced_response += "\n\n**Note**: This claim lacks scientific evidence. "
            enhanced_response += "It's important to verify health and medical information with trusted sources like WHO, CDC, or peer-reviewed medical journals. "
            enhanced_response += "Following unverified claims can be harmful. Would you like to see reliable sources about this topic?"
            
            return enhanced_response
            
        else:
            logger.info(f"✅ Applying LOW RISK personalization - standard response (risk_level: {risk_level_normalized})")
            # Low risk: Simple, clear response (base response is usually sufficient)
            return base_response

# Initialize AI chatbot
chatbot = None
AI_READY = False
MODEL_ERROR = None
MODEL_TYPE = "unknown"

try:
    chatbot = AIChatbot()
    AI_READY = True
    MODEL_TYPE = chatbot.model_type if hasattr(chatbot, 'model_type') else "unknown"
    logger.info("✅ AI Chatbot initialized and ready.")
    if MODEL_TYPE == "fallback":
        logger.warning("⚠️  Using fallback model - responses may be limited or repetitive.")
        logger.warning("   Train your model or fix model loading to get full functionality.")
except Exception as e:
    chatbot = None
    AI_READY = False
    MODEL_ERROR = str(e)
    logger.error(f"❌ Failed to initialize AI: {e}")
    logger.error(f"   Model path attempted: {SCRIPT_DIR / 'models' / 'distilbert-misinfo'}")
    logger.error(f"   Server will still run but AI features will be unavailable.")
    logger.error("   To load from Hugging Face: set MISINFO_HF_REPO=org/model (optional MISINFO_HF_SUBFOLDER=checkpoint-N).")
    logger.error("   To use local fallback only: USE_FALLBACK_MODEL=1")

def generate_with_timeout(chatbot, message, risk_level=None, susceptibility_score=None, timeout_seconds=20):
    """Generate AI response with timeout using threading"""
    result = [None]
    error = [None]
    
    def target():
        try:
            result[0] = chatbot.generate_response(message, risk_level, susceptibility_score)
        except Exception as e:
            error[0] = e
    
    thread = threading.Thread(target=target)
    thread.daemon = True
    thread.start()
    thread.join(timeout_seconds)
    
    if thread.is_alive():
        # Thread is still running, timeout occurred
        return {
            "success": True,
            "response": "Analysis timed out. This text appears to be complex and requires more processing time.",
            "is_misinformation": None,
            "is_factual": None,
            "risk_level": risk_level,
            "timestamp": datetime.now().isoformat()
        }
    
    if error[0]:
        raise error[0]
    
    return result[0]

# API Routes
@app.route('/')
def home():
    response = jsonify({
        "status": "AI Server Running",
        "ai_ready": AI_READY,
        "message": "Behavior-Aware Misinformation Detector API"
    })
    # Add CORS headers explicitly
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    return response

@app.route('/api/chat', methods=['POST', 'OPTIONS'])
def chat_endpoint():
    """Main chat endpoint for browser extension"""
    # Handle preflight OPTIONS request
    if request.method == 'OPTIONS':
        response = jsonify({})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        return response
    
    if not AI_READY:
        error_msg = "AI model not loaded"
        if MODEL_ERROR:
            error_msg = f"AI model failed to load: {MODEL_ERROR}"
        
        response = jsonify({
            "success": False,
            "error": error_msg,
            "error_details": MODEL_ERROR if MODEL_ERROR else "Model directory may be missing or corrupted",
            "response": "AI service is currently unavailable. Please check the server console for details."
        })
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response
    
    try:
        data = request.get_json()
        message = data.get('message', '').strip()
        risk_level = data.get('risk_level', None)
        susceptibility_score = data.get('susceptibility_score', None)
        
        if not message:
            return jsonify({
                "success": False,
                "error": "No message provided",
                "response": "Please provide a message to analyze."
            })
        
        logger.info(f"Received request: {message[:50]}... (Risk: {risk_level}, Score: {susceptibility_score})")
        
        # Generate AI response with timeout (personalized based on risk level)
        result = generate_with_timeout(chatbot, message, risk_level, susceptibility_score, 22)
        
        logger.info(f"Sent response: Misinfo: {result.get('is_misinformation')}")
        response = jsonify(result)
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        return response
        
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}")
        response = jsonify({
            "success": False,
            "error": str(e),
            "response": "Internal server error occurred."
        })
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        return response

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    health_data = {
        "status": "healthy" if AI_READY else "unhealthy",
        "ai_ready": AI_READY,
        "model_type": MODEL_TYPE,
        "timestamp": datetime.now().isoformat()
    }
    
    # Include error message if model failed to load
    if not AI_READY and MODEL_ERROR:
        health_data["error"] = MODEL_ERROR
        health_data["error_type"] = "model_loading_failed"
    elif AI_READY and MODEL_TYPE == "fallback":
        health_data["warning"] = "Using fallback model - limited functionality"
    
    response = jsonify(health_data)
    # Add CORS headers explicitly
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    return response

if __name__ == '__main__':
    print("Starting AI Server...")
    print("Server will run at: http://localhost:5000")
    print("Available endpoints:")
    print("   - GET  /              - Server status")
    print("   - POST /api/chat      - Main chat endpoint") 
    print("   - GET  /api/health    - Health check")
    print(f"\nModel path: {SCRIPT_DIR / 'models' / 'distilbert-misinfo'}")
    print(f"Training data: {SCRIPT_DIR / 'training_data_processed.json'}")

    app.run(host='0.0.0.0', port=5000, debug=False)