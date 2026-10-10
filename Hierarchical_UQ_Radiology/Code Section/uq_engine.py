import numpy as np

class HierarchicalUQ:
    """
    Core engine for Hierarchical Uncertainty Quantification.
    Calculates word, sentence, and section level confidence scores.
    """
    def __init__(self, model, num_passes=30, dropout_rate=0.25):
        self.model = model
        self.num_passes = num_passes
        self.dropout_rate = dropout_rate

    def get_word_uq(self, sentence_tokens, visual_features):
        """Perform MC Dropout passes to get word-level statistics."""
        word_samples = [[] for _ in sentence_tokens]
        
        for _ in range(self.num_passes):
            probs = self.model.decode_tokens(visual_features, sentence_tokens, 
                                            dropout_active=True, 
                                            dropout_rate=self.dropout_rate)
            for i, p in enumerate(probs):
                word_samples[i].append(p)
        
        word_results = []
        for i, word in enumerate(sentence_tokens):
            samples = np.array(word_samples[i])
            mean_conf = np.mean(samples)
            std_dev = np.std(samples)
            # Predictive Entropy (approximated)
            entropy = - (mean_conf * np.log(mean_conf) + (1 - mean_conf) * np.log(1 - mean_conf))
            
            word_results.append({
                "word": word,
                "confidence": mean_conf,
                "variance": std_dev,
                "entropy": entropy
            })
        return word_results

    def calculate_hierarchy(self, report_structure, visual_features):
        """Calculates multi-level scores across the whole report."""
        final_hierarchy = []
        
        for section_name, sentences in report_structure.items():
            section_sentences = []
            for sent_tokens in sentences:
                word_data = self.get_word_uq(sent_tokens, visual_features)
                
                # Sentence Confidence: Geometric Mean of Word Confidences
                # Penalizes statements with even one highly uncertain word
                word_confs = [w['confidence'] for w in word_data]
                sentence_conf = np.exp(np.mean(np.log(word_confs)))
                
                section_sentences.append({
                    "text": " ".join(sent_tokens),
                    "confidence": sentence_conf,
                    "word_data": word_data
                })
            
            # Section Confidence: Arithmetic Mean of Sentence Confidences
            section_conf = np.mean([s['confidence'] for s in section_sentences])
            
            final_hierarchy.append({
                "section": section_name,
                "sentences": section_sentences,
                "confidence": section_conf
            })
            
        return final_hierarchy
