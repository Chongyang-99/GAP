import os
import sys
import numpy as np
import zarr
import shutil
import argparse
import cv2
import h5py
import torch

GAP_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, GAP_ROOT)
sys.path.insert(0, os.path.join(GAP_ROOT, "thirdparty"))
from pi3.models.pi3 import Pi3

from gap_policy.model.vision.dinov3_encoder import DINOV3


PRETRAINED_ROOT = os.environ.get("GAP_PRETRAINED_ROOT", "pretrained")
DEFAULT_PI3_PATH = os.path.join(PRETRAINED_ROOT, "Pi3")
DEFAULT_DINOV3_WEIGHTS = os.path.join(
    PRETRAINED_ROOT,
    "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
)


def resolve_repo_path(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(GAP_ROOT, path)


def get_temporal_indices(current_idx, observation_chunk, interval):
    """
    Get temporal indices to sample for the current frame.

    Args:
        current_idx: Current frame index
        observation_chunk: Size of temporal window to look back
        interval: Interval for sampling frames

    Returns:
        List of frame indices to sample (including current frame)

    Examples:
        - current_idx=25, observation_chunk=20, interval=5
          -> [5, 10, 15, 20, 25] (5 frames)
        - current_idx=15, observation_chunk=20, interval=5
          -> uniformly sample from [0, 15]
        - current_idx=3, observation_chunk=20, interval=5
          -> [0, 1, 2, 3] (all available frames)
    """
    if observation_chunk == 0:
        # No temporal context, only use current frame
        return [current_idx]

    # Calculate number of frames to sample
    num_frames = observation_chunk // interval + 1

    if current_idx >= observation_chunk:
        # Sufficient history: sample with fixed interval
        indices = [current_idx - observation_chunk + i * interval for i in range(num_frames)]
    else:
        # Insufficient history: uniformly sample from available frames
        available_frames = current_idx + 1
        if available_frames <= num_frames:
            # Use all available frames
            indices = list(range(available_frames))
        else:
            # Uniformly sample num_frames from [0, current_idx]
            indices = np.linspace(0, current_idx, num_frames, dtype=int).tolist()

    return indices


def load_hdf5(dataset_path, camera_names=None):
    """Load data from HDF5 file (following DP pattern)"""
    if not os.path.isfile(dataset_path):
        print(f"Dataset does not exist at \n{dataset_path}\n")
        exit()

    with h5py.File(dataset_path, "r") as root:
        # Load joint action vector (agent positions)
        vector = root["/joint_action/vector"][()]

        # Load RGB images from all cameras (encoded as JPEG bytes)
        image_dict = {}
        available_cameras = list(root["/observation/"].keys())

        if camera_names is None:
            camera_names = available_cameras

        for cam_name in camera_names:
            if cam_name in available_cameras:
                image_dict[cam_name] = root[f"/observation/{cam_name}/rgb"][()]
            else:
                print(f"Warning: Camera {cam_name} not found. Available: {available_cameras}")

    return vector, image_dict


def extract_pi3_features_batch(model, images_batch, device, target_size=None, verbose=False):
    """
    Extract Pi3 point-token features from batched multi-view images.

    Args:
        model: Pi3 model
        images_batch: List of list of images [[N_views], [N_views], ...] shape [B, N_views, H, W, 3] in BGR format
        device: torch device
        target_size: Tuple (H, W) for resizing. If None, auto-compute to be divisible by 14
        verbose: Whether to print resize info

    Returns:
        features: [B, N_views, num_patches, 1024] numpy array
    """
    B = len(images_batch)
    N_views = len(images_batch[0])

    # Determine target size (must be divisible by patch_size=14)
    if target_size is None:
        h, w = images_batch[0][0].shape[:2]
        target_h = round(h / 14) * 14
        target_w = round(w / 14) * 14
        target_h = max(target_h, 14)
        target_w = max(target_w, 14)
        target_size = (target_h, target_w)

    if verbose:
        print(f"[Pi3] Resizing images from {images_batch[0][0].shape[:2]} to {target_size}")

    # Convert images to RGB, resize, and normalize
    imgs_tensor_list = []
    for b in range(B):
        imgs_per_timestep = []
        for n in range(N_views):
            img = images_batch[b][n]
            img_resized = cv2.resize(img, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            img_t = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
            imgs_per_timestep.append(img_t)
        imgs_tensor_list.append(torch.stack(imgs_per_timestep, dim=0))

    # Stack: [B, N, 3, H, W]
    imgs_tensor = torch.stack(imgs_tensor_list, dim=0).to(device)

    B, N, C, H, W = imgs_tensor.shape
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    all_point_hidden = []

    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=dtype):
            for b in range(B):
                imgs_input = imgs_tensor[b:b+1]  # [1, N, 3, H, W]
                imgs_input = (imgs_input - model.image_mean) / model.image_std

                imgs_flat = imgs_input.reshape(N, C, H, W)
                hidden = model.encoder(imgs_flat, is_training=True)
                if isinstance(hidden, dict):
                    hidden = hidden["x_norm_patchtokens"]

                hidden, pos = model.decode(hidden, N, H, W)
                point_hidden = model.point_decoder(hidden, xpos=pos)  # [N, hw, 1024]
                all_point_hidden.append(point_hidden[:, model.patch_start_idx:].float())  # [N, num_patches, 1024]

    return torch.stack(all_point_hidden, dim=0).cpu().numpy()


def extract_dinov3_features_batch(model, images_batch, device):
    """
    Extract DinoV3 features from batched multi-view images

    Args:
        model: DINOV3 model
        images_batch: List of list of images [[N_views], [N_views], ...] shape [B, N_views, H, W, 3] in BGR format
        device: torch device

    Returns:
        features: [B, N_views, num_patches, embed_dim] numpy array
    """
    # ImageNet normalization stats
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

    B = len(images_batch)
    N_views = len(images_batch[0])

    # Convert images to RGB tensor and normalize
    all_features = []

    with torch.no_grad():
        for b in range(B):
            imgs_tensor = []
            for n in range(N_views):
                img = images_batch[b][n]
                # BGR to RGB
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # To tensor [3, H, W] and normalize to [0, 1]
                img_t = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
                imgs_tensor.append(img_t)

            # Stack: [N, 3, H, W]
            imgs_tensor = torch.stack(imgs_tensor, dim=0).to(device)

            # Apply ImageNet normalization
            imgs_tensor = (imgs_tensor - mean) / std

            # Extract features
            features = model(imgs_tensor)  # [N, num_patches, embed_dim]
            all_features.append(features)

    # Stack all batches
    features_batch = torch.stack(all_features, dim=0).cpu().numpy()  # [B, N, num_patches, embed_dim]

    return features_batch


def main():
    parser = argparse.ArgumentParser(
        description="Process robot demonstration data with Pi3 and DinoV3 features"
    )
    parser.add_argument(
        "task_name",
        type=str,
        help="The name of the task (e.g., beat_block_hammer)",
    )
    parser.add_argument(
        "task_config", 
        type=str, 
        help="Task configuration name"
    )
    parser.add_argument(
        "expert_data_num",
        type=int,
        help="Number of episodes to process (e.g., 50)",
    )
    parser.add_argument(
        "--cameras",
        type=str,
        nargs="+",
        default=["head_camera"],
        help="Camera names for multi-view features",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size for feature extraction (number of frames to process together)",
    )
    parser.add_argument(
        "--model_3d",
        type=str,
        choices=["pi3"],
        default="pi3",
        help="3D backbone model used for feature extraction",
    )
    parser.add_argument(
        "--observation_chunk",
        type=int,
        default=20,
        help="Temporal window size to look back from current frame (0 means only current frame)",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=5,
        help="Interval for sampling frames within observation_chunk",
    )
    parser.add_argument(
        "--raw_data_root",
        type=str,
        default=os.environ.get("RAW_DATA_ROOT", "./data/raw"),
        help="Root directory containing RoboTwin raw task data",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=os.environ.get("OUTPUT_ROOT", "./data"),
        help="Directory for generated zarr datasets",
    )
    parser.add_argument(
        "--pi3_model_name_or_path",
        type=str,
        default=os.environ.get("PI3_MODEL_NAME_OR_PATH", os.environ.get("PI3_MODEL_PATH", DEFAULT_PI3_PATH)),
        help="Local Pi3 model directory",
    )
    parser.add_argument(
        "--dinov3_repo_dir",
        type=str,
        default=os.environ.get("DINOV3_REPO_DIR", "thirdparty/dinov3"),
        help="Local DINOv3 repository directory",
    )
    parser.add_argument(
        "--dinov3_weights_path",
        type=str,
        default=os.environ.get(
            "DINOV3_WEIGHTS_PATH",
            DEFAULT_DINOV3_WEIGHTS,
        ),
        help="DINOv3 checkpoint path",
    )
    args = parser.parse_args()

    task_name = args.task_name
    num = args.expert_data_num
    task_config = args.task_config

    load_dir = os.path.join(args.raw_data_root, str(task_name), str(task_config))
    save_dir = os.path.join(
        args.output_root,
        f"{task_name}-{task_config}-{num}-{args.model_3d}-{args.observation_chunk}-{args.interval}.zarr",
    )

    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(os.path.dirname(save_dir), exist_ok=True)

    # Load models
    device = torch.device("cuda:0")
    pi3_model_name_or_path = resolve_repo_path(args.pi3_model_name_or_path)
    dinov3_repo_dir = resolve_repo_path(args.dinov3_repo_dir)
    dinov3_weights_path = resolve_repo_path(args.dinov3_weights_path)

    print("Loading Pi3 model...")
    model_3d = Pi3.from_pretrained(pi3_model_name_or_path).to(device).eval()
    print("Pi3 model loaded successfully!")

    print(f"Loading DinoV3 model...")
    dinov3_model = DINOV3(
        model_name="dinov3_vitl16",
        freeze=True,
        repo_dir=dinov3_repo_dir,
        weights_path=dinov3_weights_path,
    ).to(device).eval()
    print(f"DinoV3 model loaded successfully!")

    total_count = 0
    current_ep = 0

    # Initialize zarr structure
    zarr_root = zarr.group(save_dir)
    zarr_data = zarr_root.create_group("data")
    zarr_meta = zarr_root.create_group("meta")

    # Storage arrays
    head_camera_arrays = []
    features_3d_arrays = []
    dinov3_features_arrays = []  # Multi-view dinov3 features
    state_arrays = []
    action_arrays = []
    episode_ends_arrays = []

    print(f"Processing with cameras: {args.cameras}")
    print(f"Batch size: {args.batch_size}")
    print(f"3D backbone: {args.model_3d}")
    print(f"Observation chunk: {args.observation_chunk}, Interval: {args.interval}")

    # Calculate number of temporal frames to sample
    if args.observation_chunk > 0:
        num_temporal_frames = args.observation_chunk // args.interval + 1
        print(f"Number of temporal frames per observation: {num_temporal_frames}")
    else:
        num_temporal_frames = 1
        print("No temporal context (using current frame only)")

    num_base_cameras = len(args.cameras)

    batch_images = []  # Buffer for batched temporal images
    batch_states = []  # Buffer for batched states
    batch_head_imgs = []  # Buffer for head camera images (current frame only)

    def process_batch():
        """Process accumulated batch of temporal images"""
        if len(batch_images) == 0:
            return

        # Extract 3D features using temporal context
        # batch_images: [B, N_temporal_views, H, W, 3]
        # where N_temporal_views = num_temporal_frames * num_cameras
        features_3d_batch = extract_pi3_features_batch(
            model_3d, batch_images, device, verbose=False
        )

        # Extract DinoV3 features only from current frame (no temporal context needed)
        # batch_current_frame_images: [B, N_cameras, H, W, 3]
        batch_current_frame_images = []
        for temporal_images_list in batch_images:
            # Extract only current frame for each camera
            current_frame_images = []
            for cam_idx in range(num_base_cameras):
                # Last temporal frame for each camera is the current frame
                view_idx = cam_idx * num_temporal_frames + (num_temporal_frames - 1)
                current_frame_images.append(temporal_images_list[view_idx])
            batch_current_frame_images.append(current_frame_images)

        dinov3_features_batch = extract_dinov3_features_batch(
            dinov3_model, batch_current_frame_images, device
        )

        # features_3d_batch shape: [B, N_temporal_views, num_patches, embed_dim]
        # We only want features from current frame (last temporal frame for each camera)
        # Temporal views are organized as: [cam0_t0, cam0_t1, ..., cam1_t0, cam1_t1, ..., camN_t0, camN_t1, ...]
        # We want: [cam0_tN, cam1_tN, ..., camN_tN] (last temporal frame for each camera)

        for i in range(len(batch_images)):
            # Extract 3D features only from current frame (last temporal frame for each camera)
            current_frame_features_3d = []

            for cam_idx in range(num_base_cameras):
                # Each camera has num_temporal_frames views
                # Get the last one (current frame) for this camera
                view_idx = cam_idx * num_temporal_frames + (num_temporal_frames - 1)
                current_frame_features_3d.append(features_3d_batch[i][view_idx])

            # Stack features from all cameras at current frame
            current_frame_features_3d = np.stack(current_frame_features_3d, axis=0)  # [num_cameras, num_patches, embed_dim]
            # DinoV3 features already from current frame only: [num_cameras, num_patches, embed_dim]
            current_frame_features_dinov3 = dinov3_features_batch[i]

            head_camera_arrays.append(batch_head_imgs[i])
            features_3d_arrays.append(current_frame_features_3d)
            dinov3_features_arrays.append(current_frame_features_dinov3)
            state_arrays.append(batch_states[i])

        # Clear batch buffers
        batch_images.clear()
        batch_states.clear()
        batch_head_imgs.clear()

    while current_ep < num:
        print(f"processing episode: {current_ep + 1} / {num}", end="\r")

        load_path = os.path.join(load_dir, f"data/episode{current_ep}.hdf5")
        vector_all, image_dict_all = load_hdf5(load_path, camera_names=args.cameras)

        # Decode all images for this episode first
        episode_images = {}  # {cam_name: [frames]}
        for cam_name in args.cameras:
            if cam_name in image_dict_all:
                frames = []
                for j in range(len(image_dict_all[cam_name])):
                    img_bit = image_dict_all[cam_name][j]
                    img = cv2.imdecode(np.frombuffer(img_bit, np.uint8), cv2.IMREAD_COLOR)
                    frames.append(img)
                episode_images[cam_name] = frames
            else:
                print(f"\nWarning: Camera {cam_name} not found in episode {current_ep}")

        num_frames_in_episode = vector_all.shape[0]
        for j in range(num_frames_in_episode):
            
            joint_state = vector_all[j]

            if j != num_frames_in_episode - 1:
                if j < args.observation_chunk:
                    temporal_indices = get_temporal_indices(j, args.observation_chunk, args.interval)
                    temporal_images_list = []
                    for cam_name in args.cameras:
                        if cam_name in episode_images:
                            for t_idx in temporal_indices:
                                temporal_images_list.append(episode_images[cam_name][t_idx])
                        else:
                            for _ in temporal_indices:
                                temporal_images_list.append(np.zeros_like(episode_images[args.cameras[0]][0]))

                    features_3d_batch = extract_pi3_features_batch(
                        model_3d, [temporal_images_list], device, verbose=False
                    )

                    current_frame_images = []
                    for cam_idx in range(num_base_cameras):
                        view_idx = cam_idx * len(temporal_indices) + (len(temporal_indices) - 1)
                        current_frame_images.append(temporal_images_list[view_idx])

                    dinov3_features_batch = extract_dinov3_features_batch(
                        dinov3_model, [current_frame_images], device
                    )

                    current_frame_features_3d = []
                    for cam_idx in range(num_base_cameras):
                        view_idx = cam_idx * len(temporal_indices) + (len(temporal_indices) - 1)
                        current_frame_features_3d.append(features_3d_batch[0][view_idx])

                    current_frame_features_3d = np.stack(current_frame_features_3d, axis=0)
                    current_frame_features_dinov3 = dinov3_features_batch[0]

                    head_camera_arrays.append(episode_images[args.cameras[0]][j])
                    features_3d_arrays.append(current_frame_features_3d)
                    dinov3_features_arrays.append(current_frame_features_dinov3)
                    state_arrays.append(joint_state)

                else:
                    temporal_indices = get_temporal_indices(j, args.observation_chunk, args.interval)
                    temporal_images_list = []
                    for cam_name in args.cameras:
                        if cam_name in episode_images:
                            for t_idx in temporal_indices:
                                temporal_images_list.append(episode_images[cam_name][t_idx])
                        else:
                            for _ in temporal_indices:
                                temporal_images_list.append(np.zeros_like(episode_images[args.cameras[0]][0]))

                    batch_images.append(temporal_images_list)
                    batch_states.append(joint_state)
                    batch_head_imgs.append(episode_images[args.cameras[0]][j])

                    if len(batch_images) >= args.batch_size:
                        process_batch()

            if j!=0:
                action_arrays.append(joint_state)

        # Process remaining batch at episode end
        if len(batch_images) > 0:
            process_batch()

        current_ep += 1
        total_count += vector_all.shape[0] - 1
        episode_ends_arrays.append(total_count)

    print(f"Total frames: {total_count}")

    episode_ends_arrays = np.array(episode_ends_arrays)
    state_arrays = np.array(state_arrays)
    head_camera_arrays = np.array(head_camera_arrays)
    features_3d_arrays = np.array(features_3d_arrays)  # [T, N_views, num_patches, embed_dim]
    dinov3_features_arrays = np.array(dinov3_features_arrays)  # [T, N_views, num_patches, embed_dim]
    action_arrays = np.array(action_arrays)

    head_camera_arrays = np.moveaxis(head_camera_arrays, -1, 1)

    compressor = zarr.Blosc(cname="zstd", clevel=3, shuffle=1)

    state_chunk_size = (100, state_arrays.shape[1])
    action_chunk_size = (100, action_arrays.shape[1])
    head_camera_chunk_size = (100, *head_camera_arrays.shape[1:])
    features_3d_chunk_size = (100, *features_3d_arrays.shape[1:])
    dinov3_features_chunk_size = (100, *dinov3_features_arrays.shape[1:])

    # Use the 3D model name in the dataset key
    features_3d_key = f"{args.model_3d}_features"

    zarr_data.create_dataset(
        "head_camera",
        data=head_camera_arrays,
        chunks=head_camera_chunk_size,
        overwrite=True,
        compressor=compressor,
    )
    zarr_data.create_dataset(
        features_3d_key,
        data=features_3d_arrays,
        chunks=features_3d_chunk_size,
        dtype="float32",
        overwrite=True,
        compressor=compressor,
    )
    zarr_data.create_dataset(
        "dinov3_features",
        data=dinov3_features_arrays,
        chunks=dinov3_features_chunk_size,
        dtype="float32",
        overwrite=True,
        compressor=compressor,
    )
    # Note: 3D depth/world points can be saved separately if needed
    zarr_data.create_dataset(
        "state",
        data=state_arrays,
        chunks=state_chunk_size,
        dtype="float32",
        overwrite=True,
        compressor=compressor,
    )
    zarr_data.create_dataset(
        "action",
        data=action_arrays,
        chunks=action_chunk_size,
        dtype="float32",
        overwrite=True,
        compressor=compressor,
    )
    zarr_meta.create_dataset(
        "episode_ends",
        data=episode_ends_arrays,
        dtype="int64",
        overwrite=True,
        compressor=compressor,
    )

    print(f"Data saved to: {save_dir}")
    print(f"  head_camera: {head_camera_arrays.shape}")
    print(f"  {features_3d_key}: {features_3d_arrays.shape}")
    print(f"  dinov3_features: {dinov3_features_arrays.shape}")
    print(f"  state: {state_arrays.shape}")
    print(f"  action: {action_arrays.shape}")
    print(f"  episode_ends: {episode_ends_arrays.shape}")

if __name__ == "__main__":
    main()
