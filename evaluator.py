# evaluator.py

import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os
import time
import yaml
from einops import rearrange


def amplitude2rssi(amplitude):
    """
    Convert normalized amplitude back to RSSI (matches dataloader)
    """
    if torch.is_tensor(amplitude):
        amplitude = amplitude.numpy()
    
    # Linear conversion (matches dataloader.py)
    rssi_db = -100 * (1 - amplitude)
    return rssi_db


class ComprehensiveEvaluator:
    """
    Comprehensive evaluation for NeRF2 RSSI prediction
    """
    
    def __init__(self, logger, logdir, expname, devices):
        self.logger = logger
        self.logdir = logdir
        self.expname = expname
        self.devices = devices
        self.results_dir = os.path.join(logdir, expname, "evaluation")
        os.makedirs(self.results_dir, exist_ok=True)
    
    def eval_network_rssi_comprehensive(self, nerf2_network, renderer, test_iter, 
                                gateway_positions=None, save_plots=True):
        """
        Comprehensive evaluation with multiple metrics and visualizations
        """
        self.logger.info("="*70)
        self.logger.info("COMPREHENSIVE EVALUATION START")
        self.logger.info("="*70)
        
        nerf2_network.eval()
        
        # Collect all predictions and ground truth
        all_predictions = []
        all_ground_truth = []
        all_positions = []
        
        import time
        inference_times = []

        with torch.no_grad():
            for test_input, test_label in test_iter:
                test_input, test_label = test_input.to(self.devices), test_label.to(self.devices)
                tx_o, rays_o, rays_d = test_input[:, :3], test_input[:, 3:6], test_input[:, 6:]
             
                # Time inference
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                start_time = time.time()

                # Predict RSSI
                predict_amplitude = renderer.render_rssi(tx_o, rays_o, rays_d)
                
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                batch_time = time.time() - start_time
                
                # Record per-sample time
                batch_size = len(test_input)
                per_sample_time = batch_time / batch_size
                inference_times.extend([per_sample_time] * batch_size)

                # RSSI calculation from amplitude
                predict_rssi = amplitude2rssi(predict_amplitude.detach().cpu())
                gt_rssi = amplitude2rssi(test_label.detach().cpu())
                
                # Convert to numpy if needed
                if torch.is_tensor(predict_rssi):
                    predict_rssi = predict_rssi.numpy()
                if torch.is_tensor(gt_rssi):
                    gt_rssi = gt_rssi.numpy()
                
                # Store results
                all_predictions.append(predict_rssi)
                all_ground_truth.append(gt_rssi)
                all_positions.append(tx_o.detach().cpu().numpy())
        
        # Concatenate all batches and flatten
        predictions = np.concatenate(all_predictions).flatten()
        ground_truth = np.concatenate(all_ground_truth).flatten()
        positions = np.concatenate(all_positions)
        
        self.logger.info(f"Total samples: {len(predictions)}")
        self.logger.info(f"Predictions shape: {predictions.shape}")
        self.logger.info(f"Ground truth shape: {ground_truth.shape}")
        self.logger.info(f"Positions shape: {positions.shape}")
        
        # Check for shape mismatch
        if len(predictions) != len(ground_truth):
            self.logger.error(f"Shape mismatch! Predictions: {len(predictions)}, Ground truth: {len(ground_truth)}")
            raise ValueError("Predictions and ground truth have different lengths")
        
        # Remove floor values (-100 dB) for meaningful metrics
        valid_mask = ground_truth > -100
        predictions_valid = predictions[valid_mask]
        ground_truth_valid = ground_truth[valid_mask]
        
        self.logger.info(f"Valid samples (RSSI > -100 dB): {len(predictions_valid)} ({100*len(predictions_valid)/len(predictions):.1f}%)")
        
        if len(predictions_valid) == 0:
            self.logger.error("No valid predictions found! All RSSI values are at floor (-100 dB)")
            return None
        
        # ====================================================================
        # 1. RSSI PREDICTION ACCURACY METRICS
        # ====================================================================
        self.logger.info("\n" + "="*70)
        self.logger.info("1. RSSI PREDICTION ACCURACY")
        self.logger.info("="*70)
        
        metrics = self._compute_rssi_metrics(ground_truth_valid, predictions_valid)
        
        self.logger.info(f"Mean Absolute Error (MAE):        {metrics['mae']:.2f} dB")
        self.logger.info(f"Root Mean Squared Error (RMSE):   {metrics['rmse']:.2f} dB")
        self.logger.info(f"Median Absolute Error:            {metrics['median_error']:.2f} dB")
        self.logger.info(f"Standard Deviation of Errors:     {metrics['std']:.2f} dB")
        self.logger.info(f"Mean Error (Bias):                {metrics['mean_error']:.2f} dB")
        self.logger.info(f"Pearson Correlation:              {metrics['correlation']:.4f}")
        self.logger.info(f"R² Score:                         {metrics['r2']:.4f}")
        self.logger.info(f"25th Percentile Error:            {metrics['p25']:.2f} dB")
        self.logger.info(f"75th Percentile Error:            {metrics['p75']:.2f} dB")
        self.logger.info(f"90th Percentile Error:            {metrics['p90']:.2f} dB")
        self.logger.info(f"95th Percentile Error:            {metrics['p95']:.2f} dB")
        
        # ====================================================================
        # 2. ERROR DISTRIBUTION ANALYSIS
        # ====================================================================
        self.logger.info("\n" + "="*70)
        self.logger.info("2. ERROR DISTRIBUTION ANALYSIS")
        self.logger.info("="*70)
        
        errors = predictions_valid - ground_truth_valid
        abs_errors = np.abs(errors)
        
        # Bins for error histogram
        error_bins = [0, 2, 5, 10, 15, np.inf]
        error_labels = ['<2 dB', '2-5 dB', '5-10 dB', '10-15 dB', '>15 dB']
        
        for i in range(len(error_bins)-1):
            lower, upper = error_bins[i], error_bins[i+1]
            count = np.sum((abs_errors >= lower) & (abs_errors < upper))
            percentage = 100 * count / len(abs_errors)
            self.logger.info(f"Errors {error_labels[i]:<10}: {count:6d} ({percentage:5.1f}%)")
        
        # ====================================================================
        # 3. COVERAGE/DETECTION ACCURACY
        # ====================================================================
        self.logger.info("\n" + "="*70)
        self.logger.info("3. COVERAGE/DETECTION ACCURACY")
        self.logger.info("="*70)
        
        thresholds = [-85, -90, -95]
        for thresh in thresholds:
            coverage_metrics = self._compute_coverage_metrics(
                ground_truth, predictions, threshold=thresh
            )
            self.logger.info(f"\nThreshold: {thresh} dB")
            self.logger.info(f"  Detection Accuracy: {coverage_metrics['accuracy']:.1%}")
            self.logger.info(f"  Precision:          {coverage_metrics['precision']:.3f}")
            self.logger.info(f"  Recall:             {coverage_metrics['recall']:.3f}")
            self.logger.info(f"  F1 Score:           {coverage_metrics['f1']:.3f}")
            self.logger.info(f"  False Positives:    {coverage_metrics['false_positive']}")
            self.logger.info(f"  False Negatives:    {coverage_metrics['false_negative']}")
        
        # ====================================================================
        # 4. SIGNAL STRENGTH RANGE ANALYSIS
        # ====================================================================
        self.logger.info("\n" + "="*70)
        self.logger.info("4. PERFORMANCE BY SIGNAL STRENGTH")
        self.logger.info("="*70)
        
        rssi_ranges = [(-100, -80), (-80, -70), (-70, -60), (-60, -50)]
        range_labels = ['Very Weak', 'Weak', 'Moderate', 'Strong']
        
        for (lower, upper), label in zip(rssi_ranges, range_labels):
            mask = (ground_truth_valid >= lower) & (ground_truth_valid < upper)
            if np.sum(mask) > 0:
                range_mae = np.mean(np.abs(predictions_valid[mask] - ground_truth_valid[mask]))
                range_count = np.sum(mask)
                self.logger.info(f"{label:<12} [{lower:4d}, {upper:4d} dB): "
                            f"MAE = {range_mae:5.2f} dB, N = {range_count:5d}")
        
        # ====================================================================
        # 5. SAVE DETAILED RESULTS
        # ====================================================================
        self._save_results(predictions, ground_truth, positions, metrics)
        
        # ====================================================================
        # 6. GENERATE VISUALIZATIONS
        # ====================================================================
        if save_plots:
            self.logger.info("\n" + "="*70)
            self.logger.info("6. GENERATING VISUALIZATIONS")
            self.logger.info("="*70)
            
            try:
                self._plot_error_distribution(ground_truth_valid, predictions_valid)
                self._plot_prediction_scatter(ground_truth_valid, predictions_valid)
                self._plot_cdf(ground_truth_valid, predictions_valid)
                self._plot_error_vs_signal_strength(ground_truth_valid, predictions_valid)
                
                # Try spatial plotting if positions are available
                if positions.shape[1] >= 2:
                    self._plot_spatial_errors(positions, predictions, ground_truth)
                
                self.logger.info("✓ All plots saved to: " + self.results_dir)
            except Exception as e:
                self.logger.error(f"Error generating plots: {e}")
                import traceback
                traceback.print_exc()
        
        # ====================================================================
        # SUMMARY
        # ====================================================================
        self.logger.info("\n" + "="*70)
        self.logger.info("EVALUATION SUMMARY")
        self.logger.info("="*70)
        self.logger.info(f"Total test samples:        {len(predictions)}")
        self.logger.info(f"Valid samples (>-100 dB):  {len(predictions_valid)}")
        self.logger.info(f"Overall MAE:               {metrics['mae']:.2f} dB")
        self.logger.info(f"Overall Median Error:      {metrics['median_error']:.2f} dB")
        self.logger.info(f"Correlation:               {metrics['correlation']:.4f}")

        # Performance rating
        if metrics['mae'] < 2.0:
            rating = "EXCELLENT ⭐⭐⭐⭐⭐"
        elif metrics['mae'] < 3.5:
            rating = "VERY GOOD ⭐⭐⭐⭐"
        elif metrics['mae'] < 5.0:
            rating = "GOOD ⭐⭐⭐"
        elif metrics['mae'] < 8.0:
            rating = "ACCEPTABLE ⭐⭐"
        else:
            rating = "NEEDS IMPROVEMENT ⭐"

        self.logger.info(f"Performance Rating:        {rating}")
        self.logger.info("="*70)

        # ====================================================================
        # 7. TIMING ANALYSIS & COMPARISON
        # ====================================================================
        inference_times_arr = np.array(inference_times)
        timing_metrics = self._compute_timing_metrics(inference_times_arr)
        
        # Save timing to metrics
        metrics['timing'] = timing_metrics
        
        # Display timing and comparison
        self._display_timing_analysis(timing_metrics, save_plots)
        
        return metrics


    def _compute_timing_metrics(self, inference_times):
        """
        Compute timing statistics
        """
        return {
            'mean_ms_per_pair': float(np.mean(inference_times) * 1000),
            'median_ms_per_pair': float(np.median(inference_times) * 1000),
            'std_ms': float(np.std(inference_times) * 1000),
            'min_ms': float(np.min(inference_times) * 1000),
            'max_ms': float(np.max(inference_times) * 1000),
            'p95_ms': float(np.percentile(inference_times, 95) * 1000),
            'throughput_pairs_per_sec': float(1.0 / np.mean(inference_times)),
        }


    def _display_timing_analysis(self, timing_metrics, save_plots):
        """
        Display timing analysis and comparison with ray-tracing
        """
        import yaml
        
        self.logger.info("\n" + "="*70)
        self.logger.info("INFERENCE TIMING ANALYSIS")
        self.logger.info("="*70)
        
        self.logger.info(f"Mean time per TX-gateway pair: {timing_metrics['mean_ms_per_pair']:.2f} ms")
        self.logger.info(f"Median time per pair:          {timing_metrics['median_ms_per_pair']:.2f} ms")
        self.logger.info(f"Std dev:                       {timing_metrics['std_ms']:.2f} ms")
        self.logger.info(f"Min time:                      {timing_metrics['min_ms']:.2f} ms")
        self.logger.info(f"Max time:                      {timing_metrics['max_ms']:.2f} ms")
        self.logger.info(f"95th percentile:               {timing_metrics['p95_ms']:.2f} ms")
        self.logger.info(f"Throughput:                    {timing_metrics['throughput_pairs_per_sec']:.2f} pairs/second")
        self.logger.info("="*70)
        
        # Save NeRF2 timing
        timing_file = os.path.join(self.results_dir, "nerf2_timing.yml")
        with open(timing_file, 'w') as f:
            yaml.dump(timing_metrics, f, default_flow_style=False)
        self.logger.info(f"✓ NeRF2 timing saved to: {timing_file}")
        
        # Load and compare with ray-tracing
        raytracing_timing_file = "../ray_tracing_dataset_gen/result/raytracing_timing.yml"
        if not os.path.exists(raytracing_timing_file):
            self.logger.warning(f"Ray-tracing timing file not found: {raytracing_timing_file}")
            self.logger.warning("Skipping comparison. Run ray-tracing with timing enabled.")
            return
        
        with open(raytracing_timing_file, 'r') as f:
            rt_timing = yaml.safe_load(f)
        
        # Check if per-pair timing exists
        if 'mean_time_per_pair_ms' not in rt_timing:
            self.logger.warning("Ray-tracing timing file missing per-pair metrics.")
            self.logger.warning("Please regenerate ray-tracing timing with updated script.")
            return
        
        # ====================================================================
        # COMPARISON: PER TX-GATEWAY PAIR
        # ====================================================================
        self.logger.info("\n" + "="*70)
        self.logger.info("COMPARISON: PER TX-GATEWAY PAIR")
        self.logger.info("="*70)
        
        rt_time_per_pair = rt_timing['mean_time_per_pair_ms']
        nerf2_time_per_pair = timing_metrics['mean_ms_per_pair']
        
        speed_ratio = rt_time_per_pair / nerf2_time_per_pair
        
        self.logger.info(f"Ray-Tracing per pair:          {rt_time_per_pair:.2f} ms")
        self.logger.info(f"NeRF2 per pair:                {nerf2_time_per_pair:.2f} ms")
        
        if speed_ratio > 1.0:
            self.logger.info(f"Speed ratio:                   {speed_ratio:.2f}x (NeRF2 is faster ⚡)")
        else:
            self.logger.info(f"Speed ratio:                   {speed_ratio:.3f}x (Ray-tracing is {1/speed_ratio:.2f}x faster)")
        
        self.logger.info(f"Time difference:               {nerf2_time_per_pair - rt_time_per_pair:.2f} ms")
        
        # Throughput comparison
        rt_throughput_pairs = 1000.0 / rt_time_per_pair
        nerf2_throughput_pairs = timing_metrics['throughput_pairs_per_sec']
        
        self.logger.info(f"\nRay-Tracing throughput:        {rt_throughput_pairs:.2f} pairs/second")
        self.logger.info(f"NeRF2 throughput:              {nerf2_throughput_pairs:.2f} pairs/second")
        
        # ====================================================================
        # COMPARISON: MULTI-GATEWAY SCENARIO
        # ====================================================================
        self.logger.info("\n" + "="*70)
        self.logger.info("COMPARISON: MULTI-GATEWAY SYSTEM PERFORMANCE")
        self.logger.info("="*70)
        
        num_gateways = rt_timing.get('num_gateways', 21)
        rt_time_per_tx = rt_timing['mean_time_per_sample_ms']
        nerf2_time_per_tx_sequential = nerf2_time_per_pair * num_gateways
        
        system_speed_ratio = rt_time_per_tx / nerf2_time_per_tx_sequential
        
        self.logger.info(f"Number of gateways:            {num_gateways}")
        self.logger.info(f"Ray-Tracing (parallel):        {rt_time_per_tx:.2f} ms for {num_gateways} gateways")
        self.logger.info(f"NeRF2 (sequential):            {nerf2_time_per_tx_sequential:.2f} ms for {num_gateways} gateways")
        
        if system_speed_ratio > 1.0:
            self.logger.info(f"System-level ratio:            {system_speed_ratio:.2f}x (NeRF2 faster)")
        else:
            self.logger.info(f"System-level ratio:            {system_speed_ratio:.3f}x (Ray-tracing {1/system_speed_ratio:.2f}x faster)")
        
        # Throughput for full TX queries
        rt_throughput_tx = 1000.0 / rt_time_per_tx
        nerf2_throughput_tx = 1000.0 / nerf2_time_per_tx_sequential
        
        self.logger.info(f"\nRay-Tracing throughput:        {rt_throughput_tx:.2f} TX/second")
        self.logger.info(f"NeRF2 throughput:              {nerf2_throughput_tx:.2f} TX/second")
        self.logger.info("="*70)
        
        # ====================================================================
        # INTERPRETATION
        # ====================================================================
        self.logger.info("\n" + "="*70)
        self.logger.info("INTERPRETATION")
        self.logger.info("="*70)
        
        if speed_ratio > 1.0:
            self.logger.info(f"✓ NeRF2 is {speed_ratio:.1f}x faster per RSSI prediction")
            self.logger.info(f"✓ Enables real-time localization at {nerf2_throughput_pairs:.0f} queries/sec")
        else:
            self.logger.info(f"• Ray-tracing is {1/speed_ratio:.1f}x faster per RSSI prediction")
            self.logger.info(f"• Ray-tracing benefits from:")
            self.logger.info(f"  - Highly optimized GPU-accelerated kernels")
            self.logger.info(f"  - Parallel computation of all gateways ({rt_time_per_tx:.1f} ms for {num_gateways})")
            self.logger.info(f"  - Decades of computer graphics research")
            self.logger.info(f"\n• NeRF2 advantages:")
            self.logger.info(f"  - Model compression (~50 MB vs full scene geometry)")
            self.logger.info(f"  - No geometry required at inference time")
            self.logger.info(f"  - Potential for optimization (quantization, pruning, batching)")
        
        self.logger.info("="*70)
        
        # ====================================================================
        # GENERATE TIMING COMPARISON PLOTS
        # ====================================================================
        if save_plots:
            try:
                self._plot_timing_comparison(rt_timing, timing_metrics, speed_ratio, num_gateways)
                self.logger.info("✓ Timing comparison plot saved")
            except Exception as e:
                self.logger.error(f"Error generating timing comparison plot: {e}")
                import traceback
                traceback.print_exc()
    
    # ========================================================================
    # HELPER FUNCTIONS
    # ========================================================================
    
    def _compute_rssi_metrics(self, ground_truth, predictions):
        """Compute comprehensive RSSI prediction metrics"""
        errors = predictions - ground_truth
        abs_errors = np.abs(errors)
        
        # Handle edge cases
        if len(errors) == 0:
            return {key: 0.0 for key in ['mae', 'rmse', 'median_error', 'mean_error', 
                                         'std', 'p25', 'p75', 'p90', 'p95', 'correlation', 'r2']}
        
        metrics = {
            'mae': np.mean(abs_errors),
            'rmse': np.sqrt(np.mean(errors**2)),
            'median_error': np.median(abs_errors),
            'mean_error': np.mean(errors),  # Bias
            'std': np.std(errors),
            'p25': np.percentile(abs_errors, 25),
            'p75': np.percentile(abs_errors, 75),
            'p90': np.percentile(abs_errors, 90),
            'p95': np.percentile(abs_errors, 95),
        }
        
        # Correlation and R2 (handle constant predictions)
        try:
            if np.std(ground_truth) > 0 and np.std(predictions) > 0:
                metrics['correlation'] = np.corrcoef(ground_truth, predictions)[0, 1]
                ss_res = np.sum(errors**2)
                ss_tot = np.sum((ground_truth - np.mean(ground_truth))**2)
                metrics['r2'] = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            else:
                metrics['correlation'] = 0.0
                metrics['r2'] = 0.0
        except:
            metrics['correlation'] = 0.0
            metrics['r2'] = 0.0
        
        return metrics
    
    def _compute_coverage_metrics(self, ground_truth, predictions, threshold=-85):
        """Compute coverage/detection accuracy metrics"""
        detected_gt = ground_truth > threshold
        detected_pred = predictions > threshold
        
        tp = np.sum(detected_gt & detected_pred)
        fp = np.sum(~detected_gt & detected_pred)
        tn = np.sum(~detected_gt & ~detected_pred)
        fn = np.sum(detected_gt & ~detected_pred)
        
        metrics = {
            'accuracy': (tp + tn) / len(ground_truth) if len(ground_truth) > 0 else 0,
            'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'true_positive': int(tp),
            'false_positive': int(fp),
            'false_negative': int(fn),
            'true_negative': int(tn),
        }
        
        # F1 score
        if metrics['precision'] + metrics['recall'] > 0:
            metrics['f1'] = 2 * metrics['precision'] * metrics['recall'] / \
                           (metrics['precision'] + metrics['recall'])
        else:
            metrics['f1'] = 0
        
        return metrics
    
    def _per_gateway_analysis(self, ground_truth, predictions, positions, gateway_positions):
        """
        Analyze performance per gateway
        
        Note: This assumes your data structure has one prediction per gateway per sample.
        You may need to adjust based on your actual data format.
        """
        num_gateways = len(gateway_positions)
        num_samples = len(ground_truth) // num_gateways
        
        results = {}
        
        for i, gw_id in enumerate(sorted(gateway_positions.keys())):
            # Extract data for this gateway (assumes interleaved format)
            gw_indices = np.arange(i, len(ground_truth), num_gateways)
            gw_gt = ground_truth[gw_indices]
            gw_pred = predictions[gw_indices]
            
            # Filter valid samples
            valid = gw_gt > -100
            
            if np.sum(valid) > 0:
                gw_gt_valid = gw_gt[valid]
                gw_pred_valid = gw_pred[valid]
                
                results[gw_id] = {
                    'mae': np.mean(np.abs(gw_gt_valid - gw_pred_valid)),
                    'rmse': np.sqrt(np.mean((gw_gt_valid - gw_pred_valid)**2)),
                    'detection_rate': np.sum(valid) / len(valid),
                    'num_samples': int(np.sum(valid))
                }
        
        return results
    
    def _save_results(self, predictions, ground_truth, positions, metrics):
        """Save detailed results to files"""
        # Save predictions vs ground truth
        result_file = os.path.join(self.results_dir, "predictions.txt")
        with open(result_file, 'w') as f:
            f.write("GroundTruth_dB,Predicted_dB,Error_dB,AbsError_dB\n")
            for gt, pred in zip(ground_truth, predictions):
                error = pred - gt
                f.write(f"{gt:.2f},{pred:.2f},{error:.2f},{abs(error):.2f}\n")
        
        # Save metrics summary
        metrics_file = os.path.join(self.results_dir, "metrics_summary.txt")
        with open(metrics_file, 'w') as f:
            f.write("RSSI Prediction Metrics\n")
            f.write("="*50 + "\n")
            for key, value in metrics.items():
                f.write(f"{key:<20}: {value:.4f}\n")
        
        self.logger.info(f"✓ Results saved to: {result_file}")
        self.logger.info(f"✓ Metrics saved to: {metrics_file}")
    
    def _plot_error_distribution(self, ground_truth, predictions):
        """Plot error distribution histogram"""
        errors = predictions - ground_truth
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram
        axes[0].hist(errors, bins=50, edgecolor='black', alpha=0.7)
        axes[0].axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
        axes[0].axvline(np.median(errors), color='green', linestyle='--', 
                       linewidth=2, label=f'Median: {np.median(errors):.2f} dB')
        axes[0].set_xlabel('Prediction Error (dB)', fontsize=12)
        axes[0].set_ylabel('Count', fontsize=12)
        axes[0].set_title('Error Distribution', fontsize=14)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Absolute error histogram
        abs_errors = np.abs(errors)
        axes[1].hist(abs_errors, bins=50, edgecolor='black', alpha=0.7, color='orange')
        axes[1].axvline(np.median(abs_errors), color='green', linestyle='--', 
                       linewidth=2, label=f'Median: {np.median(abs_errors):.2f} dB')
        axes[1].set_xlabel('Absolute Error (dB)', fontsize=12)
        axes[1].set_ylabel('Count', fontsize=12)
        axes[1].set_title('Absolute Error Distribution', fontsize=14)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'error_distribution.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_prediction_scatter(self, ground_truth, predictions):
        """Plot predictions vs ground truth scatter"""
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Scatter plot
        ax.scatter(ground_truth, predictions, alpha=0.3, s=10)
        
        # Perfect prediction line
        min_val = min(ground_truth.min(), predictions.min())
        max_val = max(ground_truth.max(), predictions.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, 
               label='Perfect Prediction')
        
        # Error bands
        ax.fill_between([min_val, max_val], 
                       [min_val - 5, max_val - 5],
                       [min_val + 5, max_val + 5],
                       alpha=0.2, color='green', label='±5 dB')
        
        ax.set_xlabel('Ground Truth RSSI (dB)', fontsize=12)
        ax.set_ylabel('Predicted RSSI (dB)', fontsize=12)
        ax.set_title('Predictions vs Ground Truth', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'prediction_scatter.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_cdf(self, ground_truth, predictions):
        """Plot CDF of absolute errors"""
        abs_errors = np.abs(predictions - ground_truth)
        sorted_errors = np.sort(abs_errors)
        cdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(sorted_errors, cdf, linewidth=2)
        
        # Reference lines
        for percentile, color in [(50, 'green'), (67, 'blue'), (90, 'orange'), (95, 'red')]:
            value = np.percentile(abs_errors, percentile)
            ax.axhline(percentile/100, color=color, linestyle='--', alpha=0.7)
            ax.axvline(value, color=color, linestyle='--', alpha=0.7,
                      label=f'P{percentile}: {value:.2f} dB')
        
        ax.set_xlabel('Absolute Error (dB)', fontsize=12)
        ax.set_ylabel('Cumulative Probability', fontsize=12)
        ax.set_title('CDF of Absolute Errors', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'error_cdf.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_error_vs_signal_strength(self, ground_truth, predictions):
        """Plot error vs signal strength"""
        errors = predictions - ground_truth
        abs_errors = np.abs(errors)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Signed error vs RSSI
        axes[0].scatter(ground_truth, errors, alpha=0.3, s=10)
        axes[0].axhline(0, color='red', linestyle='--', linewidth=2)
        axes[0].set_xlabel('Ground Truth RSSI (dB)', fontsize=12)
        axes[0].set_ylabel('Prediction Error (dB)', fontsize=12)
        axes[0].set_title('Error vs Signal Strength', fontsize=14)
        axes[0].grid(True, alpha=0.3)
        
        # Absolute error vs RSSI
        axes[1].scatter(ground_truth, abs_errors, alpha=0.3, s=10)
        axes[1].set_xlabel('Ground Truth RSSI (dB)', fontsize=12)
        axes[1].set_ylabel('Absolute Error (dB)', fontsize=12)
        axes[1].set_title('Absolute Error vs Signal Strength', fontsize=14)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'error_vs_rssi.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_spatial_errors(self, positions, predictions, ground_truth):
        """Plot spatial distribution of errors"""
        errors = np.abs(predictions - ground_truth)
        
        # Filter valid samples
        valid = ground_truth > -100
        pos_valid = positions[valid]
        errors_valid = errors[valid]
        
        if pos_valid.shape[1] < 2:
            self.logger.warning("Cannot plot spatial errors: positions must have at least 2 dimensions")
            return
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        scatter = ax.scatter(pos_valid[:, 0], pos_valid[:, 1], 
                           c=errors_valid, cmap='viridis', 
                           s=20, alpha=0.6, vmin=0, vmax=np.percentile(errors_valid, 95))
        
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Absolute Error (dB)', fontsize=12)
        
        ax.set_xlabel('X Position (m)', fontsize=12)
        ax.set_ylabel('Y Position (m)', fontsize=12)
        ax.set_title('Spatial Distribution of Prediction Errors', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'spatial_errors.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()


    def _plot_timing_comparison(self, rt_timing, nerf2_timing, speed_ratio, num_gateways):
        import matplotlib.pyplot as plt
        import os

        rt_color = '#ff6b6b'
        nerf2_color = '#51cf66'

        # ======================================================
        # Plot 1: Per-Pair Inference Time
        # ======================================================
        fig, ax1 = plt.subplots(figsize=(8,5))

        methods = ['Ray-Tracing', 'NeRF2']
        times = [rt_timing['mean_time_per_pair_ms'], nerf2_timing['mean_ms_per_pair']]
        colors = [rt_color, nerf2_color]

        bars = ax1.bar(methods, times, color=colors, edgecolor='black', linewidth=2, alpha=0.8)
        ax1.set_ylabel('Time per TX-Gateway Pair (ms)', fontsize=13, fontweight='bold')
        ax1.set_title('Per-Pair Inference Time Comparison', fontsize=15, fontweight='bold', pad=15)
        ax1.grid(True, alpha=0.3, axis='y', linestyle='--')
        ax1.set_ylim(0, max(times) * 1.2)

        for bar, time in zip(bars, times):
            height = bar.get_height()
            ax1.text(bar.get_x()+bar.get_width()/2, height,
                    f'{time:.2f} ms', ha='center', va='bottom',
                    fontweight='bold', fontsize=12)

        plt.savefig(os.path.join(self.results_dir, 'timing_per_pair.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # ======================================================
        # Plot 2: Speed Ratio
        # ======================================================
        fig, ax2 = plt.subplots(figsize=(8,5))

        if speed_ratio > 1.0:
            label_text = f'NeRF2 is {speed_ratio:.2f}x Faster'
            color = nerf2_color
        else:
            label_text = f'Ray-Tracing is {1/speed_ratio:.2f}x Faster'
            color = rt_color

        bar = ax2.bar(['Speed Ratio'], [max(speed_ratio, 1/speed_ratio)],
                    color=color, edgecolor='black', linewidth=2, alpha=0.8, width=0.5)

        ax2.set_ylabel('Speed Ratio', fontsize=13, fontweight='bold')
        ax2.set_title(label_text, fontsize=15, fontweight='bold', pad=15)
        ax2.grid(True, alpha=0.3, axis='y', linestyle='--')

        ax2.text(0, max(speed_ratio,1/speed_ratio),
                f'{max(speed_ratio,1/speed_ratio):.2f}x',
                ha='center', va='bottom', fontweight='bold', fontsize=14)

        ax2.axhline(y=1, color='gray', linestyle='--', linewidth=2, alpha=0.7, label='Equal speed (1x)')
        ax2.legend(loc='upper right', fontsize=10)

        plt.savefig(os.path.join(self.results_dir, 'speed_ratio.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # ======================================================
        # Plot 3: Multi-Gateway System Performance
        # ======================================================
        fig, ax3 = plt.subplots(figsize=(8,5))

        rt_time_per_tx = rt_timing['mean_time_per_sample_ms']
        nerf2_time_per_tx = nerf2_timing['mean_ms_per_pair'] * num_gateways

        methods = ['Ray-Tracing', 'NeRF2']
        times_tx = [rt_time_per_tx, nerf2_time_per_tx]
        colors = [rt_color, nerf2_color]

        bars = ax3.bar(methods, times_tx, color=colors, edgecolor='black', linewidth=2, alpha=0.8)

        ax3.set_ylabel(f'Time per TX ({num_gateways} gateways) [ms]', fontsize=13, fontweight='bold')
        ax3.set_title('Multi-Gateway System Performance', fontsize=15, fontweight='bold', pad=15)
        ax3.grid(True, alpha=0.3, axis='y', linestyle='--')
        ax3.set_ylim(0, max(times_tx)*1.15)

        for bar, time in zip(bars, times_tx):
            height = bar.get_height()
            ax3.text(bar.get_x()+bar.get_width()/2, height,
                    f'{time:.1f} ms',
                    ha='center', va='bottom', fontweight='bold', fontsize=11)

        plt.savefig(os.path.join(self.results_dir, 'system_performance.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # ======================================================
        # Plot 4: Throughput Comparison
        # ======================================================
        fig, ax4 = plt.subplots(figsize=(8,5))

        rt_throughput_pair = 1000.0 / rt_timing['mean_time_per_pair_ms']
        nerf2_throughput_pair = nerf2_timing['throughput_pairs_per_sec']

        rt_throughput_tx = 1000.0 / rt_timing['mean_time_per_sample_ms']
        nerf2_throughput_tx = 1000.0 / (nerf2_timing['mean_ms_per_pair'] * num_gateways)

        metrics_names = ['Pairs/sec', 'TX/sec']
        rt_values = [rt_throughput_pair, rt_throughput_tx]
        nerf2_values = [nerf2_throughput_pair, nerf2_throughput_tx]

        x = np.arange(len(metrics_names))
        width = 0.35

        bars1 = ax4.bar(x-width/2, rt_values, width, label='Ray-Tracing',
                        color=rt_color, edgecolor='black', linewidth=1.5, alpha=0.8)
        bars2 = ax4.bar(x+width/2, nerf2_values, width, label='NeRF2',
                        color=nerf2_color, edgecolor='black', linewidth=1.5, alpha=0.8)

        ax4.set_ylabel('Throughput', fontsize=13, fontweight='bold')
        ax4.set_title('Throughput Comparison', fontsize=15, fontweight='bold', pad=15)
        ax4.set_xticks(x)
        ax4.set_xticklabels(metrics_names, fontsize=11)
        ax4.legend(loc='upper right', fontsize=11, framealpha=0.9)
        ax4.grid(True, alpha=0.3, axis='y', linestyle='--')

        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax4.text(bar.get_x()+bar.get_width()/2, height,
                        f'{height:.1f}',
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

        plt.savefig(os.path.join(self.results_dir, 'throughput_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()



# -*- coding: utf-8 -*-
"""
Beamforming accuracy and latency evaluator for NeRF2 CSI model.
Compares predicted CSI from NeRF2 against ray-tracing ground truth
using Top-1 beam prediction accuracy and inference latency.
"""




# ============================================================
# DFT CODEBOOK
# ============================================================

def generate_dft_codebook(bs_positions, oversampling=2):
    """Generate DFT beamforming codebook from antenna array geometry.

    Parameters
    ----------
    bs_positions : np.ndarray. [n_elements, 3]. Antenna element positions (m).
    oversampling : int. Oversampling factor for codebook resolution.

    Returns
    -------
    codebook : np.ndarray. [n_beams, n_elements]. Complex steering vectors.
    """
    n_elements = len(bs_positions)

    # Determine grid dimensions from unique x/y positions
    unique_x = np.unique(bs_positions[:, 0])
    unique_y = np.unique(bs_positions[:, 1])
    n_cols = len(unique_x)   # azimuth dimension
    n_rows = len(unique_y)   # elevation dimension

    N_az = n_cols * oversampling
    N_el = n_rows * oversampling

    # Normalized spatial frequencies spanning [-1, 1)
    psi_az = np.linspace(-1, 1, N_az, endpoint=False)
    psi_el = np.linspace(-1, 1, N_el, endpoint=False)

    codebook = []
    for el in psi_el:
        for az in psi_az:
            # Phase shift per element based on position
            phase = 2 * np.pi * (bs_positions[:, 0] * az + bs_positions[:, 1] * el)
            steering = np.exp(1j * phase) / np.sqrt(n_elements)
            codebook.append(steering)

    codebook = np.array(codebook)   # [n_beams, n_elements]
    return codebook


def load_bs_positions(bs_yml_path):
    """Load base station antenna element positions from yml.

    Parameters
    ----------
    bs_yml_path : str. Path to base_station.yml

    Returns
    -------
    positions : np.ndarray. [n_elements, 3].
    """
    with open(bs_yml_path, 'r') as f:
        data = yaml.safe_load(f)
    positions = np.array(data['base_station'], dtype=np.float32)
    return positions


# ============================================================
# BEAM SELECTION
# ============================================================

def select_best_beam(csi, codebook):
    """Select best beam index for each sample using maximum beamforming gain.

    Parameters
    ----------
    csi      : np.ndarray. [n_samples, n_elements, n_subcarriers]. Complex CSI.
    codebook : np.ndarray. [n_beams, n_elements]. Complex steering vectors.

    Returns
    -------
    beam_indices : np.ndarray. [n_samples]. Best beam index per sample.
    beam_gains   : np.ndarray. [n_samples]. Best beam gain per sample.
    all_gains    : np.ndarray. [n_samples, n_beams]. Gain for every beam.
    """
    # Beamforming gain: |w^H h|^2 averaged over subcarriers
    # einsum: (n_beams, n_elements) x (n_samples, n_elements, n_subcarriers)
    # -> (n_samples, n_beams, n_subcarriers)
    gains = np.abs(np.einsum('be,nes->nbs', codebook.conj(), csi)) ** 2
    gains = gains.mean(axis=-1)          # [n_samples, n_beams] avg over subcarriers
    beam_indices = np.argmax(gains, axis=-1)  # [n_samples]
    beam_gains = gains[np.arange(len(gains)), beam_indices]
    return beam_indices, beam_gains, gains


# ============================================================
# MAIN EVALUATOR
# ============================================================

class BeamformingEvaluator:

    def __init__(self, logger, logdir, expname, bs_yml_path, oversampling=2):
        """
        Parameters
        ----------
        logger      : logging.Logger
        logdir      : str. Directory for saving results.
        expname     : str. Experiment name.
        bs_yml_path : str. Path to base_station.yml.
        oversampling: int. Codebook oversampling factor.
        """
        self.logger = logger
        self.results_dir = os.path.join(logdir, expname, "beamforming_eval")
        os.makedirs(self.results_dir, exist_ok=True)

        # Load array geometry and generate codebook
        self.bs_positions = load_bs_positions(bs_yml_path)
        self.codebook = generate_dft_codebook(self.bs_positions, oversampling)
        self.n_beams = len(self.codebook)
        self.n_elements = len(self.bs_positions)

        self.logger.info(f"Array: {self.n_elements} antenna elements")
        self.logger.info(f"Codebook: {self.n_beams} beams (oversampling={oversampling})")


    def eval_beamforming(self, nerf2_network, renderer, test_iter,
                         test_set, devices, save_plots=True):
        """
        Full beamforming accuracy and latency evaluation.

        Parameters
        ----------
        nerf2_network : NeRF2 model.
        renderer      : Renderer with render_csi method.
        test_iter     : DataLoader for test set.
        test_set      : CSI_dataset instance (for denormalize_csi).
        devices       : torch device.
        save_plots    : bool. Whether to save plots.
        """
        self.logger.info("=" * 70)
        self.logger.info("BEAMFORMING EVALUATION START")
        self.logger.info("=" * 70)

        nerf2_network.eval()

        all_pred_csi = []
        all_gt_csi   = []
        inference_times = []

        # --------------------------------------------------------
        # 1. Run inference and collect CSI
        # --------------------------------------------------------
        with torch.no_grad():
            for test_input, test_label in test_iter:
                test_input = test_input.to(devices)
                test_label = test_label.to(devices)

                uplink, rays_o, rays_d = (
                    test_input[:, :52],
                    test_input[:, 52:55],
                    test_input[:, 55:]
                )

                # Time inference
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t0 = time.time()

                predict_downlink = renderer.render_csi(uplink, rays_o, rays_d)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                elapsed = time.time() - t0

                batch_size = test_input.shape[0]
                per_sample_time = elapsed / batch_size
                inference_times.extend([per_sample_time] * batch_size)

                # Reconstruct complex CSI
                gt_downlink = test_label[:, :26] + 1j * test_label[:, 26:]
                pred_downlink = predict_downlink  # [B, 26] complex

                # Denormalize
                pred_downlink = test_set.denormalize_csi(pred_downlink)
                gt_downlink   = test_set.denormalize_csi(gt_downlink)

                all_pred_csi.append(pred_downlink.cpu().numpy())
                all_gt_csi.append(gt_downlink.cpu().numpy())

        # Stack: [n_test_samples, 26] complex
        pred_csi_flat = np.concatenate(all_pred_csi, axis=0)
        gt_csi_flat   = np.concatenate(all_gt_csi,   axis=0)

        # Reshape to [n_tx, n_elements, n_subcarriers]
        # n_test_samples = n_tx * n_bs_elements
        n_bs = test_set.n_bs
        n_tx = len(pred_csi_flat) // n_bs

        pred_csi = rearrange(pred_csi_flat, '(n g) c -> n g c', g=n_bs)  # [n_tx, 8, 26]
        gt_csi   = rearrange(gt_csi_flat,   '(n g) c -> n g c', g=n_bs)  # [n_tx, 8, 26]

        self.logger.info(f"Test samples: {n_tx} TX positions x {n_bs} antenna elements")
        self.logger.info(f"CSI shape — GT: {gt_csi.shape}, Pred: {pred_csi.shape}")

        # --------------------------------------------------------
        # 2. SNR (channel prediction accuracy)
        # --------------------------------------------------------
        self.logger.info("\n" + "=" * 70)
        self.logger.info("CHANNEL PREDICTION ACCURACY (SNR)")
        self.logger.info("=" * 70)

        snr_per_sample = self._compute_snr(pred_csi, gt_csi)
        self.logger.info(f"Median SNR:     {np.median(snr_per_sample):.2f} dB")
        self.logger.info(f"Mean SNR:       {np.mean(snr_per_sample):.2f} dB")
        self.logger.info(f"10th pct SNR:   {np.percentile(snr_per_sample, 10):.2f} dB")
        self.logger.info(f"90th pct SNR:   {np.percentile(snr_per_sample, 90):.2f} dB")

        # --------------------------------------------------------
        # 3. Beam prediction accuracy
        # --------------------------------------------------------
        self.logger.info("\n" + "=" * 70)
        self.logger.info("BEAM PREDICTION ACCURACY")
        self.logger.info("=" * 70)

        gt_beams,   gt_gains,   gt_all_gains   = select_best_beam(gt_csi,   self.codebook)
        pred_beams, pred_gains, pred_all_gains = select_best_beam(pred_csi, self.codebook)

        # Top-1 accuracy
        top1_acc = np.mean(gt_beams == pred_beams)

        # Top-3 accuracy: predicted beam is within top-3 GT beams
        top3_acc = self._topk_accuracy(gt_all_gains, pred_beams, k=3)
        top5_acc = self._topk_accuracy(gt_all_gains, pred_beams, k=5)

        self.logger.info(f"Top-1 Beam Accuracy:  {top1_acc:.2%}")
        self.logger.info(f"Top-3 Beam Accuracy:  {top3_acc:.2%}")
        self.logger.info(f"Top-5 Beam Accuracy:  {top5_acc:.2%}")
        self.logger.info(f"Total beams in codebook: {self.n_beams}")

        # Beamforming gain loss: gain using predicted beam vs optimal beam
        # actual_gain = gain you get using NeRF2-selected beam on GT channel
        actual_gains = gt_all_gains[np.arange(n_tx), pred_beams]
        gain_loss_db = 10 * np.log10(gt_gains / (actual_gains + 1e-30))

        self.logger.info(f"\nBeamforming Gain Loss (vs optimal):")
        self.logger.info(f"  Median: {np.median(gain_loss_db):.2f} dB")
        self.logger.info(f"  Mean:   {np.mean(gain_loss_db):.2f} dB")
        self.logger.info(f"  90th pct: {np.percentile(gain_loss_db, 90):.2f} dB")

        # --------------------------------------------------------
        # 4. Latency
        # --------------------------------------------------------
        self.logger.info("\n" + "=" * 70)
        self.logger.info("INFERENCE LATENCY")
        self.logger.info("=" * 70)

        times_arr = np.array(inference_times)
        timing = {
            'mean_ms':   float(np.mean(times_arr)   * 1000),
            'median_ms': float(np.median(times_arr) * 1000),
            'std_ms':    float(np.std(times_arr)    * 1000),
            'p95_ms':    float(np.percentile(times_arr, 95) * 1000),
            'throughput_per_sec': float(1.0 / np.mean(times_arr)),
        }

        self.logger.info(f"Mean inference time:   {timing['mean_ms']:.2f} ms per sample")
        self.logger.info(f"Median inference time: {timing['median_ms']:.2f} ms per sample")
        self.logger.info(f"95th percentile:       {timing['p95_ms']:.2f} ms")
        self.logger.info(f"Throughput:            {timing['throughput_per_sec']:.1f} samples/sec")

        # Load ray-tracing timing if available
        self._compare_latency(timing)

        # --------------------------------------------------------
        # 5. Plots
        # --------------------------------------------------------
        if save_plots:
            self._plot_beam_accuracy(gt_beams, pred_beams, top1_acc, top3_acc, top5_acc)
            self._plot_gain_loss_cdf(gain_loss_db)
            self._plot_snr_cdf(snr_per_sample)
            self._plot_confusion_matrix(gt_beams, pred_beams)
            self.logger.info(f"\n✓ Plots saved to: {self.results_dir}")

        # --------------------------------------------------------
        # Summary
        # --------------------------------------------------------
        self.logger.info("\n" + "=" * 70)
        self.logger.info("SUMMARY")
        self.logger.info("=" * 70)
        self.logger.info(f"Top-1 Beam Accuracy:      {top1_acc:.2%}")
        self.logger.info(f"Median Gain Loss:          {np.median(gain_loss_db):.2f} dB")
        self.logger.info(f"Median SNR:                {np.median(snr_per_sample):.2f} dB")
        self.logger.info(f"NeRF2 Inference Time:      {timing['mean_ms']:.2f} ms/sample")

        return {
            'top1_accuracy':    float(top1_acc),
            'top3_accuracy':    float(top3_acc),
            'top5_accuracy':    float(top5_acc),
            'median_gain_loss': float(np.median(gain_loss_db)),
            'mean_gain_loss':   float(np.mean(gain_loss_db)),
            'median_snr':       float(np.median(snr_per_sample)),
            'timing':           timing,
        }


    # ============================================================
    # HELPER METHODS
    # ============================================================

    def _compute_snr(self, pred_csi, gt_csi):
        """Compute per-sample SNR: 10*log10(||h||^2 / ||h - h_pred||^2)"""
        signal_power = np.sum(np.abs(gt_csi) ** 2,   axis=(1, 2))
        noise_power  = np.sum(np.abs(gt_csi - pred_csi) ** 2, axis=(1, 2))
        snr = 10 * np.log10(signal_power / (noise_power + 1e-30))
        return snr

    def _topk_accuracy(self, gt_all_gains, pred_beams, k):
        """Fraction of samples where predicted beam is in top-k GT beams."""
        # Top-k GT beam indices per sample
        topk_gt = np.argsort(gt_all_gains, axis=-1)[:, -k:]  # [n_samples, k]
        correct = np.array([pred_beams[i] in topk_gt[i] for i in range(len(pred_beams))])
        return np.mean(correct)

    def _compare_latency(self, nerf2_timing):
        """Compare with ray-tracing timing if file exists."""
        rt_path = "../ray_tracing_dataset_gen/result/raytracing_timing.yml"
        if not os.path.exists(rt_path):
            self.logger.warning("Ray-tracing timing file not found, skipping comparison")
            return

        import yaml
        with open(rt_path, 'r') as f:
            rt = yaml.safe_load(f)

        rt_time  = rt.get('mean_time_per_pair_ms', rt.get('mean_time_per_sample_ms', None))
        if rt_time is None:
            return

        nerf2_time = nerf2_timing['mean_ms']
        speedup = rt_time / nerf2_time

        self.logger.info("\n--- Latency Comparison ---")
        self.logger.info(f"Ray-Tracing:  {rt_time:.2f} ms/sample")
        self.logger.info(f"NeRF2:        {nerf2_time:.2f} ms/sample")
        if speedup > 1:
            self.logger.info(f"Speedup:      {speedup:.2f}x  (NeRF2 faster ⚡)")
        else:
            self.logger.info(f"Speedup:      {speedup:.3f}x  (Ray-tracing {1/speedup:.2f}x faster)")

    # ============================================================
    # PLOTS
    # ============================================================

    def _plot_beam_accuracy(self, gt_beams, pred_beams, top1, top3, top5):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Left: bar chart of Top-K accuracy
        ks      = ['Top-1', 'Top-3', 'Top-5']
        accs    = [top1,    top3,    top5]
        colors  = ['#4dabf7', '#51cf66', '#ffd43b']
        bars = axes[0].bar(ks, [a * 100 for a in accs], color=colors,
                           edgecolor='black', linewidth=1.5, alpha=0.85)
        for bar, acc in zip(bars, accs):
            axes[0].text(bar.get_x() + bar.get_width()/2,
                         bar.get_height() + 1,
                         f'{acc:.1%}', ha='center', fontweight='bold', fontsize=12)
        axes[0].set_ylabel('Accuracy (%)', fontsize=12)
        axes[0].set_title('Beam Prediction Accuracy', fontsize=14, fontweight='bold')
        axes[0].set_ylim(0, 110)
        axes[0].grid(True, axis='y', alpha=0.3)

        # Right: predicted vs GT beam scatter
        axes[1].scatter(gt_beams, pred_beams, alpha=0.4, s=15, c='#4dabf7')
        axes[1].plot([0, self.n_beams], [0, self.n_beams], 'r--',
                     linewidth=2, label='Perfect prediction')
        axes[1].set_xlabel('GT Best Beam Index',   fontsize=12)
        axes[1].set_ylabel('Predicted Best Beam Index', fontsize=12)
        axes[1].set_title('Predicted vs GT Beam Index', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'beam_accuracy.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_gain_loss_cdf(self, gain_loss_db):
        sorted_loss = np.sort(gain_loss_db)
        cdf = np.arange(1, len(sorted_loss) + 1) / len(sorted_loss)

        fig, ax = plt.subplots(figsize=(9, 6))
        ax.plot(sorted_loss, cdf, linewidth=2, color='#4dabf7')

        for pct, color in [(50, 'green'), (90, 'orange'), (95, 'red')]:
            val = np.percentile(gain_loss_db, pct)
            ax.axhline(pct/100, color=color, linestyle='--', alpha=0.7)
            ax.axvline(val,     color=color, linestyle='--', alpha=0.7,
                       label=f'P{pct}: {val:.2f} dB')

        ax.set_xlabel('Beamforming Gain Loss vs Optimal (dB)', fontsize=12)
        ax.set_ylabel('CDF', fontsize=12)
        ax.set_title('CDF of Beamforming Gain Loss', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'gain_loss_cdf.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_snr_cdf(self, snr_per_sample):
        sorted_snr = np.sort(snr_per_sample)
        cdf = np.arange(1, len(sorted_snr) + 1) / len(sorted_snr)

        fig, ax = plt.subplots(figsize=(9, 6))
        ax.plot(sorted_snr, cdf, linewidth=2, color='#51cf66')

        median_snr = np.median(snr_per_sample)
        ax.axvline(median_snr, color='red', linestyle='--',
                   label=f'Median: {median_snr:.2f} dB')

        ax.set_xlabel('SNR (dB)', fontsize=12)
        ax.set_ylabel('CDF', fontsize=12)
        ax.set_title('CDF of Channel Prediction SNR', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'snr_cdf.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_confusion_matrix(self, gt_beams, pred_beams):
        """Beam confusion matrix — shows which beams get confused with which."""
        conf = np.zeros((self.n_beams, self.n_beams), dtype=int)
        for g, p in zip(gt_beams, pred_beams):
            conf[g, p] += 1

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(conf, cmap='Blues', aspect='auto')
        plt.colorbar(im, ax=ax, label='Count')
        ax.set_xlabel('Predicted Beam Index', fontsize=12)
        ax.set_ylabel('GT Beam Index',        fontsize=12)
        ax.set_title('Beam Prediction Confusion Matrix', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'beam_confusion_matrix.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()


    def eval_beamforming_from_csi(self, pred_csi, gt_csi, inference_times, save_plots=True):
        """
        Run beamforming evaluation on already-collected CSI arrays.
        Called from eval_network_csi after inference loop.

        Parameters
        ----------
        pred_csi        : np.ndarray (n_tx, n_bs, n_subcarriers) complex
        gt_csi          : np.ndarray (n_tx, n_bs, n_subcarriers) complex
        inference_times : list of float, per-sample inference time in seconds
        """
        n_tx = pred_csi.shape[0]
        self.logger.info(f"Beamforming eval: {n_tx} TX positions, {self.n_beams} beams")

        # SNR
        snr = self._compute_snr(pred_csi, gt_csi)
        self.logger.info(f"Median SNR:    {np.median(snr):.2f} dB")
        self.logger.info(f"Mean SNR:      {np.mean(snr):.2f} dB")

        # Beam selection
        gt_beams,   gt_gains,   gt_all_gains   = select_best_beam(gt_csi,   self.codebook)
        pred_beams, pred_gains, pred_all_gains = select_best_beam(pred_csi, self.codebook)

        top1 = np.mean(gt_beams == pred_beams)
        top3 = self._topk_accuracy(gt_all_gains, pred_beams, k=3)
        top5 = self._topk_accuracy(gt_all_gains, pred_beams, k=5)

        self.logger.info(f"Top-1 Accuracy: {top1:.2%}")
        self.logger.info(f"Top-3 Accuracy: {top3:.2%}")
        self.logger.info(f"Top-5 Accuracy: {top5:.2%}")

        # Gain loss
        actual_gains = gt_all_gains[np.arange(n_tx), pred_beams]
        gain_loss_db = 10 * np.log10(gt_gains / (actual_gains + 1e-30))
        self.logger.info(f"Median Gain Loss: {np.median(gain_loss_db):.2f} dB")

        # Latency
        times_arr = np.array(inference_times)
        timing = {
            'mean_ms':            float(np.mean(times_arr) * 1000),
            'median_ms':          float(np.median(times_arr) * 1000),
            'p95_ms':             float(np.percentile(times_arr, 95) * 1000),
            'throughput_per_sec': float(1.0 / np.mean(times_arr)),
        }
        self.logger.info(f"NeRF2 inference: {timing['mean_ms']:.2f} ms/sample")
        self._compare_latency(timing)

        if save_plots:
            self._plot_beam_accuracy(gt_beams, pred_beams, top1, top3, top5)
            self._plot_gain_loss_cdf(gain_loss_db)
            self._plot_snr_cdf(snr)
            self._plot_confusion_matrix(gt_beams, pred_beams)

        return {
            'top1_accuracy':    float(top1),
            'top3_accuracy':    float(top3),
            'top5_accuracy':    float(top5),
            'median_gain_loss': float(np.median(gain_loss_db)),
            'median_snr':       float(np.median(snr)),
            'timing':           timing,
        }