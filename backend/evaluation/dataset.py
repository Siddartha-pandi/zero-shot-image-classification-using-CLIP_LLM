"""Dataset management for evaluation."""

import json
from pathlib import Path
from typing import List, Dict


class EvaluationDataset:
    """Load and manage evaluation datasets in JSON format."""
    
    def __init__(self, dataset_path: str):
        self.path = Path(dataset_path)
        self.samples = self._load_dataset()
        
    def _load_dataset(self) -> List[Dict]:
        """
        Load dataset from JSON file.
        Expected format:
        [
            {
                "path": "path/to/image.jpg",
                "label": "cat",
                "domain": "natural"
            },
            ...
        ]
        """
        if not self.path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.path}")
        
        with open(self.path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Validate format
        for i, sample in enumerate(data):
            required = ['path', 'label', 'domain']
            for field in required:
                if field not in sample:
                    raise ValueError(f"Sample {i} missing required field: {field}")
        
        return data
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]
    
    def get_labels(self) -> List[str]:
        """Get unique labels in dataset."""
        return sorted(list(set(s['label'].lower() for s in self.samples)))
    
    def get_domains(self) -> List[str]:
        """Get unique domains in dataset."""
        return sorted(list(set(s['domain'] for s in self.samples)))
    
    def filter_by_domain(self, domain: str) -> List[Dict]:
        """Get samples from specific domain."""
        return [s for s in self.samples if s['domain'] == domain]
    
    def filter_by_labels(self, labels: List[str]) -> List[Dict]:
        """Get samples with specific labels."""
        label_set = set(l.lower() for l in labels)
        return [s for s in self.samples if s['label'].lower() in label_set]
