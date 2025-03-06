import numpy as np
import websockets
import pickle
import asyncio

server_ip = '60.205.13.252' # 目标服务器公网ip
Port = 3394


async def Connectcloud():
    # 发送数据
    try:
        async with websockets.connect(f"ws://{server_ip}:{Port}") as websocket:
            msg1 = 6
            msg2 = np.eye(3)
            msg = [msg1,msg2] # 设置发送给服务器的信息
            msg_bytes = pickle.dumps(msg)
            await websocket.send(msg_bytes)
            print('send finish')
            # 接收数据
            msg_bytes_recv = await websocket.recv()
            msg_recv = pickle.loads(msg_bytes_recv)
            print('the msg is',msg_recv)
            
    except asyncio.TimeoutError:
        print("Connect timeout!")
    
    return True
    
if __name__=="__main__":
    print('start')
    #下面代码一执行，客户端将和服务端建立连接。内容输入后客户端即发送此内容到服务端
    # result = asyncio.get_event_loop().run_until_complete(Connectcloud())
    result = asyncio.run(Connectcloud())
    print(result)
    # result = asyncio.run(Sendcloud())
    # print(result)
    