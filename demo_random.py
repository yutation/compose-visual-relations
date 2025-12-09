import argparse
import torch
import torch_xla.core.xla_model as xm
import torch_xla
import torch_xla.debug.profiler as xp
from models import ResNetModel


def generate_random_labels(batch_size, num_rels):
    """
    Generate random labels for inference.
    Label format: [shape, size, color, material, obj_idx_1, 
                    shape, size, color, material, obj_idx_2, relation]
    Shape: (batch_size, num_rels, 11)
    """
    labels = torch.randint(0, 8, (batch_size, num_rels, 11), dtype=torch.long)
    labels[:, :, 4] = 0  # First object index
    labels[:, :, 9] = 1  # Second object index
    labels[:, :, 10] = torch.randint(0, 6, (batch_size, num_rels))  # Relation
    return labels


def inference(model, batch_size, num_rels, im_size, step_lr, num_steps):
    """
    Generate images from random noise using Langevin dynamics.
    
    Args:
        model: Trained EBM model (already loaded and on device)
        batch_size: Number of samples to generate
        num_rels: Number of relations to compose
        im_size: Image size
        step_lr: Step size for Langevin dynamics
        num_steps: Number of sampling steps
    
    Returns:
        Generated images tensor (batch_size, 3, im_size, im_size)
    """
    device = next(model.parameters()).device
    
    # Generate random labels
    labels = generate_random_labels(batch_size, num_rels)
    
    # Initialize random image
    im = torch.rand(batch_size, 3, im_size, im_size).to(device)
    im_noise = torch.randn_like(im)
    
    # Prepare labels
    if len(labels.shape) == 2:  # Bx11 -> Bx1x11
        labels = labels[:, None]
    
    labels = labels.to(device)
    labels = torch.chunk(labels, chunks=labels.shape[1], dim=1)
    labels = [chunk.squeeze(dim=1) for chunk in labels]
    
    step_lr = step_lr / len(labels)
    
    # Langevin dynamics sampling
    xp.start_trace("./traces/demo")
    torch_xla.sync()
    
    print(f"Running inference with {num_steps} steps...")
    for i in range(num_steps):
        im_noise.normal_()
        im = im + 0.001 * im_noise
        im.requires_grad_(requires_grad=True)
        
        energy = sum([model.forward(im, y) for y in labels])
        im_grad = torch.autograd.grad([energy.sum()], [im])[0]
        
        im = im - step_lr * im_grad
        im = im.detach()
        im = torch.clamp(im, 0, 1)
        
        if (i + 1) % 20 == 0:
            print(f"  Step {i+1}/{num_steps}, Energy: {energy.mean().item():.4f}")
        torch_xla.sync()
    
    print("Done: im.shape", im.shape)
    xp.stop_trace()

    return im


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simple random inference for EBM')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('--num_rels', type=int, default=1, help='Number of relations')
    parser.add_argument('--num_steps', type=int, default=20, help='Number of sampling steps')
    args = parser.parse_args()
    
    # Device (TPU)
    device = torch_xla.device()
    print(f"Using device: {device}")
    
    # Load model directly
    checkpoint_path = './checkpoint/clevr/model_best.pth'
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    FLAGS = checkpoint['FLAGS']
    xp.start_server(9012)
    model = ResNetModel(FLAGS)
    model.load_state_dict(checkpoint['model_state_dict_0'])
    model = model.eval().to(device)
    print("Model loaded successfully")
    
    # Run inference
    results = inference(
        model=model,
        batch_size=args.batch_size,
        num_rels=args.num_rels,
        im_size=FLAGS.im_size,
        step_lr=FLAGS.step_lr,
        num_steps=args.num_steps
    )
    
    print(f"Inference complete. Results shape: {results.shape}")

