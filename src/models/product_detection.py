"""
Product Detection Module - Simplified Approach
Analyzes product co-occurrence patterns from metadata
"""

import json
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np

class ProductDetector:
    """
    Analyze product co-occurrence patterns from metadata
    Simplified approach - real implementation would use computer vision
    """
    
    def __init__(self, metadata_dir):
        self.metadata_dir = Path(metadata_dir)
        self.product_stats = {}
        self.cooccurrence = defaultdict(lambda: defaultdict(int))
        self._analyze_metadata()
    
    def _analyze_metadata(self):
        """Analyze all metadata files to build product statistics"""
        print("\n" + "="*70)
        print("PRODUCT DETECTION - METADATA ANALYSIS")
        print("="*70)
        print("Analyzing product patterns from metadata...")
        
        all_products = []
        image_count = 0
        
        for json_file in self.metadata_dir.glob("*.json"):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    bin_data = data.get('BIN_FCSKU_DATA', {})
                    
                    products_in_bin = list(bin_data.keys())
                    all_products.extend(products_in_bin)
                    
                    # Build co-occurrence matrix
                    for prod1 in products_in_bin:
                        for prod2 in products_in_bin:
                            if prod1 != prod2:
                                self.cooccurrence[prod1][prod2] += 1
                    
                    # Product stats
                    for asin, info in bin_data.items():
                        if asin not in self.product_stats:
                            self.product_stats[asin] = {
                                'count': 0,
                                'quantities': [],
                                'avg_quantity': 0
                            }
                        
                        self.product_stats[asin]['count'] += 1
                        qty = info.get('quantity', 1)
                        self.product_stats[asin]['quantities'].append(qty)
                    
                    image_count += 1
                    
                    # Progress indicator
                    if image_count % 1000 == 0:
                        print(f"  Processed {image_count:,} bins...")
                    
            except Exception as e:
                continue
        
        # Calculate averages
        for asin, stats in self.product_stats.items():
            stats['avg_quantity'] = np.mean(stats['quantities'])
        
        print(f"\n✓ Analyzed {image_count:,} bins")
        print(f"✓ Found {len(self.product_stats):,} unique products")
        print(f"✓ Total product occurrences: {len(all_products):,}")
        print("="*70)
    
    def get_top_products(self, n=50):
        """Get top N most frequent products"""
        sorted_products = sorted(
            self.product_stats.items(),
            key=lambda x: x[1]['count'],
            reverse=True
        )
        return [(asin, data['count'], data['avg_quantity']) 
                for asin, data in sorted_products[:n]]
    
    def get_product_info(self, asin):
        """Get statistics for a specific product"""
        if asin in self.product_stats:
            stats = self.product_stats[asin]
            cooccur_products = sorted(
                self.cooccurrence[asin].items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
            
            return {
                'asin': asin,
                'frequency': stats['count'],
                'avg_quantity': stats['avg_quantity'],
                'min_quantity': min(stats['quantities']),
                'max_quantity': max(stats['quantities']),
                'appears_with': [p[0] for p in cooccur_products]
            }
        return None
    
    def get_statistics(self):
        """Get overall statistics"""
        frequencies = [s['count'] for s in self.product_stats.values()]
        quantities = []
        for s in self.product_stats.values():
            quantities.extend(s['quantities'])
        
        return {
            'total_unique_products': len(self.product_stats),
            'avg_frequency': np.mean(frequencies),
            'median_frequency': np.median(frequencies),
            'products_appearing_once': sum(1 for f in frequencies if f == 1),
            'products_appearing_10plus': sum(1 for f in frequencies if f >= 10),
            'avg_quantity_per_occurrence': np.mean(quantities),
            'total_occurrences': len(quantities)
        }
    
    def save_results(self, output_file='../../outputs/product_detection_results.txt'):
        """Save analysis results to file"""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("PRODUCT DETECTION ANALYSIS RESULTS\n")
            f.write("="*70 + "\n\n")
            
            # Overall statistics
            stats = self.get_statistics()
            f.write("OVERALL STATISTICS:\n")
            f.write("-"*70 + "\n")
            for key, value in stats.items():
                f.write(f"{key.replace('_', ' ').title()}: {value}\n")
            
            f.write("\n" + "="*70 + "\n")
            f.write("TOP 50 MOST FREQUENT PRODUCTS\n")
            f.write("="*70 + "\n\n")
            
            for i, (asin, count, avg_qty) in enumerate(self.get_top_products(50), 1):
                f.write(f"{i:3d}. {asin}\n")
                f.write(f"     Appearances: {count:4d} | Avg Quantity: {avg_qty:.2f}\n")
                
                info = self.get_product_info(asin)
                if info and info['appears_with']:
                    f.write(f"     Often appears with: {', '.join(info['appears_with'][:5])}\n")
                f.write("\n")
        
        print(f"\n✓ Results saved to: {output_path}")


def main():
    """Run product detection analysis"""
    
    detector = ProductDetector("../../data/raw/metadata")
    
    # Display results
    print("\n" + "="*70)
    print("TOP 30 MOST FREQUENT PRODUCTS")
    print("="*70)
    print(f"{'Rank':<6} {'ASIN':<15} {'Count':<8} {'Avg Qty':<10}")
    print("-"*70)
    
    for i, (asin, count, avg_qty) in enumerate(detector.get_top_products(30), 1):
        print(f"{i:<6} {asin:<15} {count:<8} {avg_qty:<10.2f}")
    
    # Overall statistics
    stats = detector.get_statistics()
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    print(f"Total unique products: {stats['total_unique_products']:,}")
    print(f"Average frequency per product: {stats['avg_frequency']:.2f}")
    print(f"Products appearing only once: {stats['products_appearing_once']:,}")
    print(f"Products appearing 10+ times: {stats['products_appearing_10plus']:,}")
    print(f"Average quantity per occurrence: {stats['avg_quantity_per_occurrence']:.2f}")
    print("="*70)
    
    # Save results
    detector.save_results()
    
    print("\n✓ Product detection complete!")
    print("="*70)


if __name__ == "__main__":
    main()
