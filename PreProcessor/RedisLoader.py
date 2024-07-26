from turtle import st
import redis
import dgl
import pickle
import tqdm

class RedisLoader:
    def __init__(self, redis_host = 'localhost', redis_port=6379, dgl_file: str = 'dgl_graph', data_name: str = 'uboot'):
        self.connection_pool = redis.ConnectionPool(host=redis_host, port=redis_port, db=0)
        self.redis = redis.Redis(host=redis_host, port=redis_port, connection_pool=self.connection_pool)
        self.graphs = dgl.load_graphs(dgl_file)[0]
        self.data_name = data_name
        
    def put_graph(self, graph: dgl.DGLGraph, key: str):
        graph_bytes = pickle.dumps(graph)
        self.redis.set(key, graph_bytes)
        
    def put_graphs(self):
        bar = tqdm.tqdm(total=len(self.graphs))
        for i, graph in enumerate(self.graphs):
            bar.update(1)
            self.put_graph(graph, self.data_name + '-' + str(i))
            
    def flush_db(self):
        self.redis.flushdb()
        
if __name__ == '__main__':
    dgl_file = '/home/username/Project/plc/dataset/uboot_dataset/dgl_graphs.dgl'
    redis_loader = RedisLoader(dgl_file=dgl_file, data_name='uboot')
    redis_loader.flush_db()
    redis_loader.put_graphs()
    # redis_loader.flush_db()
        
