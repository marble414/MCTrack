"""Innovative modules for MCTrack."""
from .feature_alignment import HierarchicalGeometricFeatureAlignment
from .cross_modal_loss import ProbabilisticCrossModalConsistencyLoss
from .cross_modal_attention import AdaptiveCrossModalAttention
from .pretraining import ContrastiveSelfSupervisedPretraining
from .heterogeneous_gnn import HeterogeneousSpatioTemporalGNN
from .occlusion_completion import OcclusionAwareGraphCompletion
from .multi_scale_tgcn import MultiScaleTemporalGraphConvolution
from .efficiency_optim import EfficiencyOptimizationModule
from .training_pipeline import MultiStageTrainingPipeline
from .enhanced_mctrack import EnhancedMCTrack
from .experimental_design import ComprehensiveExperimentalDesign

__all__ = [
    'HierarchicalGeometricFeatureAlignment',
    'ProbabilisticCrossModalConsistencyLoss',
    'AdaptiveCrossModalAttention',
    'ContrastiveSelfSupervisedPretraining',
    'HeterogeneousSpatioTemporalGNN',
    'OcclusionAwareGraphCompletion',
    'MultiScaleTemporalGraphConvolution',
    'EfficiencyOptimizationModule',
    'MultiStageTrainingPipeline',
    'EnhancedMCTrack',
    'ComprehensiveExperimentalDesign',
]
