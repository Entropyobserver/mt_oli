# utils/data_loader.py

import json
import os
from typing import Dict, List, Tuple
from sklearn.model_selection import train_test_split

class DataLoader:
    def __init__(self, random_seed=42):
        self.random_seed = random_seed
    
    def load_json_data(self, file_path: str) -> List[Dict]:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    
    def save_json_data(self, data: List[Dict], file_path: str):
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def create_fixed_splits(self, data_file: str, 
                           train_ratio=0.8, val_ratio=0.1, test_ratio=0.1) -> Tuple[List, List, List]:
        
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
        
        data = self.load_json_data(data_file)
        
        train_data, temp_data = train_test_split(
            data, test_size=(1 - train_ratio), random_state=self.random_seed
        )
        
        val_size = val_ratio / (val_ratio + test_ratio)
        val_data, test_data = train_test_split(
            temp_data, test_size=(1 - val_size), random_state=self.random_seed
        )
        
        return train_data, val_data, test_data
    
    def save_fixed_splits(self, train_data: List, val_data: List, test_data: List,
                         train_file="train_fixed.json", val_file="val_fixed.json", 
                         test_file="test_fixed.json"):
        
        self.save_json_data(train_data, train_file)
        self.save_json_data(val_data, val_file)
        self.save_json_data(test_data, test_file)
        
        print(f"Fixed splits saved:")
        print(f"  Train: {len(train_data)} samples -> {train_file}")
        print(f"  Val: {len(val_data)} samples -> {val_file}")
        print(f"  Test: {len(test_data)} samples -> {test_file}")
    
    def load_fixed_splits(self, train_file="train_fixed.json", 
                         val_file="val_fixed.json", test_file="test_fixed.json") -> Tuple[List, List, List]:
        
        if not all(os.path.exists(f) for f in [train_file, val_file, test_file]):
            raise FileNotFoundError("Fixed split files not found. Run create_fixed_splits first.")
        
        train_data = self.load_json_data(train_file)
        val_data = self.load_json_data(val_file)
        test_data = self.load_json_data(test_file)
        
        return train_data, val_data, test_data
    
    def create_data_subset(self, data: List[Dict], size: int) -> List[Dict]:
        if size >= len(data):
            return data
        return data[:size]
    
    def get_data_info(self, data: List[Dict]) -> Dict:
        if not data:
            return {"size": 0, "avg_source_length": 0, "avg_target_length": 0}
        
        source_lengths = [len(item.get('source', '').split()) for item in data]
        target_lengths = [len(item.get('target', '').split()) for item in data]
        
        return {
            "size": len(data),
            "avg_source_length": sum(source_lengths) / len(source_lengths),
            "avg_target_length": sum(target_lengths) / len(target_lengths),
            "max_source_length": max(source_lengths),
            "max_target_length": max(target_lengths)
        }
