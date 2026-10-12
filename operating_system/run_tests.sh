#!/bin/bash

SERVER_FORK="./login_server_fork"
SERVER_THREAD="./login_server_thread"
CLIENT="./login_client"
STRESS="./login_stress"

echo -e "user1 123\nuser2 123" > users.txt #建立帳號檔包含兩組帳密


gcc login_server_fork.c -o $SERVER_FORK
gcc login_server_thread.c -lpthread -o $SERVER_THREAD
gcc login_client.c -o $CLIENT
gcc login_stress.c -lpthread -o $STRESS

echo "Starting fork server..."
$SERVER_FORK & #背景執行fork程式
SERVER_PID=$! #取得剛剛背景程序的PID 用於之後關閉
sleep 1

echo "Running stress test on fork server..."
for c in 10 50 100 200; do
    /usr/bin/time -v $STRESS user1 123 $c
done #time -v 可以詳細列出 CPU 時間、記憶體使用、上下文切換等統計

kill $SERVER_PID
sleep 100 #確保fork server完全關閉(因為我用同一個port)

echo "Starting thread server..."
$SERVER_THREAD &
SERVER_PID=$!
sleep 1

echo "Running stress test on thread server..."
for c in 10 50 100 200; do
    /usr/bin/time -v $STRESS user1 123 $c
done

kill $SERVER_PID
echo "Tests completed."