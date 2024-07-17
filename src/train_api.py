import numpy as np
import warnings
from deprecated import deprecated
import gzip,libpytrain,json
import json
import datetime, time, random
from kafka import KafkaProducer
from nist_dataset import mnist, letters
from tqdm import tqdm
import numpy as np
import pymysql
from tqdm import tqdm, trange

TOPIC 		= "mnist_stream"
SERVER 		= "localhost:9092"
BATCH_SIZE 	= 1024
MYSQL_HOST 	= "localhost"
MYSQL_PORT 	= 3306
MYSQL_USER 	= "root"
MYSQL_PASS 	= "weibo10302638"
MYSQL_DATABASE 	= "online_learning"

model_list = [
		"mlp_mnist", 
		"lenet_mnist",
		"vgg_letters"
	     ]

MODEL_CONF_PATH = "/data0/users/shuaishuai3/wt/t/t1_8/model/model_conf/"

class PTrain:
    def __init__(self, model_name):
        self.model_name = model_name
        self.dataset = mnist if self.model_name in ['mlp_mnist', 'lenet_mnist'] else letters
        self.trainer = libpytrain.ws_train(MODEL_CONF_PATH + model_name + ".json")
        self.log_path = "../save/" + self.model_name + ".log"
        self.batch_size = BATCH_SIZE
        self.connect = pymysql.connect(host=MYSQL_HOST, port=MYSQL_PORT, user=MYSQL_USER, password=MYSQL_PASS, db=MYSQL_DATABASE)
        if not self.connect:
            print("connetc to {} mysql failed".format(MYSQL_HOST))
        self.cursor = self.connect.cursor()

    def __exec(self, sql, args=None):
        try:
            self.cursor.execute(sql, args)  
        except Exception as e:
            print(e)

    def init_model(self):
        drop_table = """DROP TABLE {}_log""".format(self.model_name)
        create_table =  """
				CREATE TABLE IF NOT EXISTS `{}_log`(
				  `timestamp` VARCHAR(20) NOT NULL,
				  `loss` FLOAT(16, 6) NOT NULL,
				  `accuracy` FLOAT(16, 6) NOT NULL,
				  `time_cost` FLOAT(16, 6) NOT NULL
				) ENGINE=InnoDB DEFAULT CHARSET=utf8;

			""".format(self.model_name)
        self.__exec(drop_table)
        self.__exec(create_table)
        self.trainer.initializer()

    @deprecated("use `train` is better")
    def offline_train(self, epoch = 10):
        self.trainer.train_offline(epoch)

    def predict(self, flatten_x, x_size):
        self.trainer.predict(flatten_x, x_size)

    def train(self, times):
        plot_d = {"loss": [], "accuracy": [], "cost": []}
        with trange(times, ncols=0) as t:
            for i in t:
                rand_index = np.random.randint(0, len(self.dataset.train_images), self.batch_size)
                train_x = (self.dataset.train_images[rand_index].flatten() / 255.).tolist()
                train_y = (self.dataset.train_labels[rand_index].flatten() / 1.).tolist()
                t.set_description("EPOCH %i" % i)
                res = self.trainer.train_online(train_x, train_y)
                for key, value in zip(list(plot_d.keys()), res):
                    plot_d[key].append(value)
                t.set_postfix(loss=res[0], accuracy=res[1], time_cost_pre_batch=res[2])
        with open(self.log_path, "w") as f:
            json.dump(plot_d, f)
    
    def mock_producer_mnist_data(self):
        producer = KafkaProducer(bootstrap_servers=SERVER,
                                 api_version=(1, 0, 0),
                                 # security_protocol="SSL",
                                 # security_protocol="SASL_PLAINTEXT"
                                 # sasl_mechanism="PLAIN"
                                 # sasl_plain_username=config.USERNAME
                                 # sasl_plain_password=config.PASSWORD
                                 value_serializer=lambda m: json.dumps(m).encode()
                                 )
        data_size = 10 * 1024
        rand_index = np.random.randint(0, len(train_images), batch_size)
        train_x, train_y = train_images[rand_index], train_labels[rand_index]
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

if __name__ == "__main__":
    trainer = PTrain(model_list[2])
    trainer.init_model()
    #trainer.offline_train()
    trainer.train(10)
