import numpy as np
import warnings
from deprecated import deprecated
import gzip,libpytrain,json
import json
import datetime, time, random
from kafka import KafkaProducer
from dataset import mnist, letters, criteo
from tqdm import tqdm
import pymysql, sys
from tqdm import tqdm, trange

def mock_producer_mnist_data(train_data):
    producer = KafkaProducer(bootstrap_servers=SERVER,
                             api_version=(1, 0, 0),
                             # security_protocol="SSL",
                             # security_protocol="SASL_PLAINTEXT"
                             # sasl_mechanism="PLAIN"
                             # sasl_plain_username=config.USERNAME
                             # sasl_plain_password=config.PASSWORD
                             value_serializer=lambda m: json.dumps(m).encode()
                             )
    sample_size = BATCH_SIZE * 2
    rand_index = np.random.randint(0, len(train_data), sample_size)
    train_x, train_y = train_data[rand_index], train_labels[rand_index]
    for i, (x, y) in tqdm(enumerate(zip(train_x, train_y))):
        data = {
                    'ts': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'x' : ",".join(str(_) for _ in x.tolist()),
                    'y' : int(y)
               }
        future = producer.send(TOPIC, data)
        # record_metadata = future.get(timeout=10)
        time.sleep(random.random() / 200)

def plot_log(self, log):
    plt.figure(figsize=(18, 18), dpi=150)
    step = [_ + 1 for _ in range(len(log["loss"]))]
    acc, loss, time_cost = log["accuracy"], log["loss"], log["cost"]
    legend_list = ["acc", "loss", "time_cost"]
    for i, value in zip(range(3), [acc, loss, time_cost]):
        plt.subplot(311 + i)
        plt.plot(step, value, alpha=1., linewidth=2, label=legend_list[i])
        for x, y in zip(step, value):
            if x % 10 == 0 or x == 1:
                plt.text(x, y, round(y, 2), ha='center', va='bottom', fontsize=12.5)
        plt.legend()
    plt.show()
