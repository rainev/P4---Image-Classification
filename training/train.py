# training/train.py

import io
import time

import grpc
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image

from model import MnistCNN
from data_loader import get_mnist_loader
import dashboard_pb2, dashboard_pb2_grpc


def calculate_display_size(original_width, original_height):
    """
    scale images, specs (min:32x32 max: 512x512)
    """
    if original_width <= 32 or original_height <= 32:
        return original_width * 2, original_height * 2
    elif original_width > 512 or original_height > 512:
        scale_factor = min(512 / original_width, 512 / original_height)
        return int(original_width * scale_factor), int(original_height * scale_factor)
    else:
        return original_width, original_height

def encode_image_tensor(image_tensor, target_size=None):
    """
    Convert a single MNIST tensor (1, 28, 28) to PNG bytes.
    """
    img = image_tensor.squeeze().cpu().numpy() * 255.0  # [0,255]
    img = Image.fromarray(img.astype("uint8"), mode="L")  # grayscale

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def training_stream(model, train_loader, device, max_steps=400):
    """
    Generator used as the request iterator for StreamTrainingData.
    Does real training and yields a TrainingBatch per step.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    global_step = 0
    epoch_idx = 0
    model.train()

    # Calculate display size 
    original_width, original_height = 28, 28
    display_width, display_height = calculate_display_size(original_width, original_height)
    
    print(f"[Training] Image scaling: {original_width}x{original_height} → {display_width}x{display_height}")

    for epoch in range(1, 1000):  # big number; we break via max_steps
        epoch_idx = epoch
        for batch_idx, (images, labels) in enumerate(train_loader):
            global_step += 1

            images = images.to(device)   
            labels = labels.to(device) 

            # ----- forward + backward + optimize -----
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            # ----- prepare data for dashboard -----
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)  # (B,)

            batch_size = images.size(0)
            num_samples = min(16, batch_size)   # 4x4 grid

            image_bytes_list = []
            pred_labels = []
            true_labels = []

            for i in range(num_samples):
                image_bytes_list.append(
                    encode_image_tensor(
                        images[i], 
                        target_size=(display_width, display_height)
                    )
                )
                pred_labels.append(str(int(preds[i].cpu().item())))
                true_labels.append(str(int(labels[i].cpu().item())))

            yield dashboard_pb2.TrainingBatch(
                images=image_bytes_list,
                pred_labels=pred_labels,
                true_labels=true_labels,
                loss=float(loss.item()),
                step=global_step,
                epoch=epoch_idx,
                batch_size=num_samples,
                sent_timestamp_ms=int(time.time() * 1000),
                image_width=28,
                image_height=28,
            )

            if global_step % 50 == 0:
                print(f"[Training] Step {global_step} | Epoch {epoch_idx} | Loss: {loss.item():.4f}")

            if global_step >= max_steps:
                print(f"[Training] Reached max_steps={max_steps}, stopping.")
                return

def connect_to_dashboard_with_retry(max_retries=5):
    """
    Connect to dashboard with retry logic for fault tolerance.
    """
    for attempt in range(max_retries):
        try:
            channel = grpc.insecure_channel("localhost:50051")
            stub = dashboard_pb2_grpc.DashboardServiceStub(channel)
            
            # Test connection with Ping
            ping_req = dashboard_pb2.PingRequest(
                client_id="mnist_trainer",
                client_timestamp_ms=int(time.time() * 1000),
            )
            response = stub.Ping(ping_req)
            
            if response.alive:
                print(f"[Training] ✓ Connected to dashboard (attempt {attempt + 1})")
                return stub
                
        except grpc.RpcError as e:
            print(f"[Training] ⚠ Connection attempt {attempt + 1}/{max_retries} failed")
            if attempt < max_retries - 1:
                time.sleep(2)
    
    print("[Training] ✗ Could not connect to dashboard after retries.")
    return None


def main():

    print("\nConnecting to dashboard...")
    # ----- gRPC channel & stub -----
    stub = connect_to_dashboard_with_retry(max_retries=5)

    if not stub:
        print("[Training] Cannot proceed without dashboard. Exiting.")
        return

    # ----- device & model -----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Training] Using device: {device}")

    model = MnistCNN().to(device)

    # ----- data loader -----
    train_loader = get_mnist_loader(
        data_dir="./data",
        batch_size=16,
        num_workers=2,
    )

    # ----- start streaming training data -----

    max_reconnect_attempts = 3
    reconnect_delay = 5  # seconds

    for reconnect_attempt in range(max_reconnect_attempts):
        try:
            print(f"\n{'='*60}")
            print("\nStarting CNN training and streaming to dashboard...")
            print("=" * 60 + "\n")

            request_iter = training_stream(model, train_loader, device, max_steps=2000)
            batch_count = 0

            # BatchAck responses
            for ack in stub.StreamTrainingData(request_iter):
                batch_count += 1
                
                if not ack.ok:
                    print(f"[Training] Dashboard error: {ack.message}")
                
                # slow down training if dashboard gets slow
                if ack.frames_per_second > 0 and ack.frames_per_second < 30:
                    print(f"[Training] Dashboard FPS low ({ack.frames_per_second}), delaying traning")
                    time.sleep(0.05)
                
                # Log every 50 batchess
                if batch_count % 50 == 0:
                    latency_ms = int(time.time() * 1000) - ack.server_timestamp_ms
                    print(f"[Training] ✓ Batch {batch_count} | Dashboard FPS: {ack.frames_per_second} | Latency: {latency_ms}ms")
                    
            # streaming is finish, break out of reconnect loop
            print("\n" + "=" * 60)
            print("Training stream completed.")
            print("=" * 60)
            break

        except grpc.RpcError as e:
            print(f"\n[Training] ✗ gRPC error: {e.code()} - {e.details()}")
            
            if reconnect_attempt < max_reconnect_attempts - 1:
                print(f"[Training] Attempting to reconnect in {reconnect_delay} seconds...")
                time.sleep(reconnect_delay)
                
                # Try to reconnect
                stub = connect_to_dashboard_with_retry(max_retries=3)
                if stub is None:
                    print("[Training] Reconnection failed. Exiting.")
                    break
            else:
                print("[Training] Max reconnection attempts reached. Exiting.")


if __name__ == "__main__":
    main()