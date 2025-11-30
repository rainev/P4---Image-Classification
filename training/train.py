# training/train.py

import io
import time

import grpc
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image

from training.model import MnistCNN
from training.data_loader import get_mnist_loader
from services.proto import dashboard_pb2, dashboard_pb2_grpc


def encode_image_tensor(image_tensor):
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

    for epoch in range(1, 1000):  # big number; we break via max_steps
        epoch_idx = epoch
        for batch_idx, (images, labels) in enumerate(train_loader):
            global_step += 1

            images = images.to(device)   # (B, 1, 28, 28)
            labels = labels.to(device)   # (B,)

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
                image_bytes_list.append(encode_image_tensor(images[i]))
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

            if global_step >= max_steps:
                return  # stop generator after some steps to keep demo short


def main():
    # ----- gRPC channel & stub -----
    channel = grpc.insecure_channel("localhost:50051")
    stub = dashboard_pb2_grpc.DashboardServiceStub(channel)

    # ----- device & model -----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MnistCNN().to(device)

    # ----- data loader -----
    train_loader = get_mnist_loader(
        data_dir="./data",
        batch_size=16,
        num_workers=2,
    )

    # ----- start streaming training data -----
    print("▶ Starting CNN training and streaming to dashboard...")
    request_iter = training_stream(model, train_loader, device, max_steps=400)
    ack = stub.StreamTrainingData(request_iter)
    print("✔ Dashboard Ack:", ack.ok, ack.message)


if __name__ == "__main__":
    main()