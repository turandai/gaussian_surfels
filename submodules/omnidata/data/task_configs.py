####################
# Tasks
####################

task_parameters = {
    'autoencoding':{
        'out_channels': 3,

    },

    'denoising':{
        'out_channels': 3,

    },
    'colorization': {
        'out_channels': 3,

    },
    'class_object':{
        'out_channels': 1000,

    },
    'class_scene':{
        'out_channels': 365,

    },
    'depth_zbuffer':{
        'out_channels': 1,
        'mask_val': 1.0,
        'clamp_to': (0.0, 8000.0 / (2**16 - 1)) # Same as consistency
    },
    'depth_euclidean':{
        'out_channels': 1,
        'clamp_to': (0.0, 8000.0 / (2**16 - 1)) # Same as consistency
#         'mask_val': 1.0,
    },
    'edge_texture': {
        'out_channels': 1,
        'clamp_to': (0.0, 0.25)
    },
    'edge_occlusion': {
        'out_channels': 1,

    },
    'inpainting':{
        'out_channels': 3,

    },
    'keypoints3d': {
        'out_channels': 1,

    },
    'keypoints2d':{
        'out_channels': 1,

    },
    'principal_curvature':{
        'out_channels': 2,
        'mask_val': 0.0,
    },

    'reshading':{
        'out_channels': 1,

    }, 
    'normal':{
        'out_channels': 3,
#         'mask_val': 0.004,
        'mask_val': 0.502,
    },
    'mask_valid':{
        'out_channels': 1,
        'mask_val': 0.0,
    },
    'rgb':{
        'out_channels': 3,
    },
    'segment_semantic': {
        'out_channels': 17,
    },
    'segment_unsup2d': {
        'out_channels': 64,
    },
    'segment_unsup25d': {
        'out_channels': 64,
    },
    'segment_instance': {
    },
    'segment_panoptic': {
        'out_channels': 2,
    },
    'fragments': {
        'out_channels': 1
    }
}

        
PIX_TO_PIX_TASKS = ['colorization', 'edge_texture', 'edge_occlusion',  'keypoints3d', 'keypoints2d', 'reshading', 'depth_zbuffer', 'depth_euclidean', 'curvature', 'autoencoding', 'denoising', 'normal', 'inpainting', 'segment_unsup2d', 'segment_unsup25d', 'segment_semantic', 'segment_instance']
FEED_FORWARD_TASKS = ['class_object', 'class_scene', 'room_layout', 'vanishing_point']
SINGLE_IMAGE_TASKS = PIX_TO_PIX_TASKS + FEED_FORWARD_TASKS
SIAMESE_TASKS = ['fix_pose', 'jigsaw', 'ego_motion', 'point_match', 'non_fixated_pose']

