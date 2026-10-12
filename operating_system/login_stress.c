#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <pthread.h>
#include <time.h>

#define PORT 12345
#define BUF_SIZE 1024

typedef struct {
    char username[128];
    char password[128];
} credentials_t;

int success_count = 0;
pthread_mutex_t lock;//互斥鎖 上課有教過 就是讓cs只有一個人可以進入

void *login_thread(void *arg) {
    credentials_t *cred = (credentials_t*)arg;//強制轉型回原本的型態
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    struct sockaddr_in server_addr;
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(PORT);
    server_addr.sin_addr.s_addr = inet_addr("127.0.0.1");

    if(connect(sock, (struct sockaddr*)&server_addr, sizeof(server_addr)) == 0) {//若成功（返回 0） 進行後續登入動作
        char buf[BUF_SIZE];//把帳號密碼組成一個字串 然後傳給server
        sprintf(buf, "%s %s", cred->username, cred->password);
        write(sock, buf, strlen(buf));
        int n = read(sock, buf, BUF_SIZE-1);
        if(n > 0) {
            buf[n] = '\0';
            if(strcmp(buf, "OK") == 0) {
                pthread_mutex_lock(&lock);//鎖住
                success_count++;
                pthread_mutex_unlock(&lock);//解鎖
            }
        }
    }
    close(sock);
    return NULL;
}

int main(int argc, char *argv[]) {
    if(argc != 4) {
        printf("Usage: %s <username> <password> <concurrency>\n", argv[0]);
        return 1;
    }

    credentials_t cred;
    strcpy(cred.username, argv[1]);//程式名稱是argv[0] 所以從1開始
    strcpy(cred.password, argv[2]);
    int concurrency = atoi(argv[3]);//字串轉整數

    pthread_t threads[concurrency];
    pthread_mutex_init(&lock, NULL);//初始化互斥鎖

    clock_t start = clock();//clock()回傳目前CPU時間(現在樣開始計時的意思)
    for(int i=0;i<concurrency;i++)
        pthread_create(&threads[i], NULL, login_thread, &cred);
    for(int i=0;i<concurrency;i++)
        pthread_join(threads[i], NULL);//pthread_join 會阻塞呼叫它的程式 直到對應的thread結束
    clock_t end = clock();

    double duration = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Concurrency: %d, Success: %d, Time: %.3f sec\n", concurrency, success_count, duration);

    pthread_mutex_destroy(&lock);//釋放互斥鎖
    return 0;
}