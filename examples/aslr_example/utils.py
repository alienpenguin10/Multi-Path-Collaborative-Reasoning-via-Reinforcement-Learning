"""
Utility functions and callbacks for M3PO/ASLR training.
"""

import torch
from transformers import TrainerCallback


class M3POLoggingCallback(TrainerCallback):
    """
    Custom callback to log M3PO/ASLR-specific metrics to TensorBoard.
    Tracks router behavior, path statistics, and gating decisions during training.
    """
    def __init__(self):
        super().__init__()
        self.router_stats = []
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called when the trainer logs metrics. Extracts M3PO stats and writes to TensorBoard."""
        if logs is None:
            return
        
        # Get the model (unwrap if needed)
        model = kwargs.get('model', None)
        if model is None:
            return
            
        try:
            # Access the base model through PEFT wrapper
            base_model = model.model.model if hasattr(model.model, 'model') else model.model
            
            if not hasattr(base_model, 'm3po_router'):
                return
            
            # Collect router statistics
            with torch.no_grad():
                # Get router gate weights and bias
                gate_weight = base_model.m3po_router.gate.weight.data
                gate_bias = base_model.m3po_router.gate.bias.data
                
                # Log router parameters
                logs['m3po/router_weight_norm'] = gate_weight.norm().item()
                logs['m3po/router_bias'] = gate_bias.item()
                
                # Check if we have accumulated stats from generation
                if hasattr(base_model, '_m3po_router_stats') and base_model._m3po_router_stats['weights']:
                    # Use actual router decisions from generation
                    router_weights = base_model._m3po_router_stats['weights']
                    activations = base_model._m3po_router_stats['activations']
                    
                    logs['m3po/router_output_mean_actual'] = sum(router_weights) / len(router_weights)
                    logs['m3po/router_output_std_actual'] = torch.tensor(router_weights).std().item()
                    logs['m3po/router_output_min_actual'] = min(router_weights)
                    logs['m3po/router_output_max_actual'] = max(router_weights)
                    logs['m3po/activation_rate_actual'] = sum(activations) / len(activations)
                    logs['m3po/num_generation_steps'] = len(router_weights)
                    
                    # Clear stats for next logging period
                    base_model._m3po_router_stats = {'weights': [], 'activations': []}
                else:
                    # Simulate router output distribution for a random input
                    dummy_input = torch.randn(100, base_model.m3po_router.gate.in_features, 
                                             device=gate_weight.device, dtype=gate_weight.dtype)
                    router_outputs = torch.sigmoid(base_model.m3po_router.gate(dummy_input))
                    
                    logs['m3po/router_output_mean'] = router_outputs.mean().item()
                    logs['m3po/router_output_std'] = router_outputs.std().item()
                    logs['m3po/router_output_min'] = router_outputs.min().item()
                    logs['m3po/router_output_max'] = router_outputs.max().item()
                    
                    # Activation rate (% of tokens that would trigger "thinking")
                    threshold = base_model.m3po_router.threshold if hasattr(base_model.m3po_router, 'threshold') else 0.5
                    activation_rate = (router_outputs > threshold).float().mean().item()
                    logs['m3po/activation_rate'] = activation_rate
                
                # Log collaborator and aggregator norms
                collab_norm = sum(p.norm().item() for p in base_model.m3po_path_collaborator.parameters())
                agg_norm = sum(p.norm().item() for p in base_model.m3po_path_aggregator.parameters())
                
                logs['m3po/collaborator_weight_norm'] = collab_norm
                logs['m3po/aggregator_weight_norm'] = agg_norm
                
                # Log ASLR config for reference
                logs['m3po/aslr_num_paths'] = base_model.aslr_num_paths
                logs['m3po/aslr_max_iterations'] = base_model.aslr_max_iterations
                
                # Write directly to TensorBoard (on_log is called AFTER trainer writes)
                if hasattr(args, 'logging_dir'):
                    from torch.utils.tensorboard import SummaryWriter
                    writer = SummaryWriter(log_dir=args.logging_dir)
                    
                    # Write all M3PO metrics
                    for key in logs.keys():
                        if 'm3po' in key:
                            writer.add_scalar(key, logs[key], state.global_step)
                    
                    writer.flush()
                    writer.close()
                
        except Exception as e:
            print(f"[M3PO Logging] Error logging M3PO metrics: {e}")
            import traceback
            traceback.print_exc()
