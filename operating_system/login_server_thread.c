#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <pthread.h>

#define PORT 12345
#define BUF_SIZE 1024
#define USER_FILE "users.txt"

//跟fork版本一樣在檢查帳密是否正確
int check_credentials(const char *username, const char *password) {
    FILE *fp = fopen(USER_FILE, "r");
    if (!fp) return 0;
    char u[128], p[128];
    int ok = 0;
    while(fscanf(fp, "%s %s", u, p) != EOF) {
        if(strcmp(u, username) == 0 && strcmp(p, password) == 0) {
            ok = 1;
            break;
        }
    }
    fclose(fp);
    return ok;
}

void *handle_client(void *arg) {
    int client_sock = *(int*)arg;
    free(arg);
    char buf[BUF_SIZE];
    int n = read(client_sock, buf, BUF_SIZE-1);
    if(n > 0) {
        buf[n] = '\0';
        char username[128], password[128];
        sscanf(buf, "%s %s", username, password);
        if(check_credentials(username, password)) {
            write(client_sock, "OK", 2);
        } else {
            write(client_sock, "FAIL", 4);
        }
    }
    close(client_sock);
    return NULL;
}

int main() {
    int server_sock;
    struct sockaddr_in server_addr, client_addr;
    socklen_t addr_len = sizeof(client_addr);

    server_sock = socket(AF_INET, SOCK_STREAM, 0);
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(PORT);
    server_addr.sin_addr.s_addr = INADDR_ANY;

    bind(server_sock, (struct sockaddr*)&server_addr, sizeof(server_addr));
    listen(server_sock, 10);

    printf("Thread server listening on port %d...\n", PORT);

    //跟fork版本不一樣的地方在這裡
    while(1) {
        int *client_sock = malloc(sizeof(int));//pthread需要傳指標 所以要配置記憶體
        *client_sock = accept(server_sock, (struct sockaddr*)&client_addr, &addr_len);
        //阻塞等待新連線到來 並且建立一個新的socket來和這個客戶端通訊
        if(*client_sock < 0) continue;//accept失敗 跳過
        pthread_t tid;
        pthread_create(&tid, NULL, handle_client, client_sock);//建立新的thread執行handle_client
        pthread_detach(tid);//分離thread 結束後自動回收資源(這樣就不會產生殭屍thread)
    }
    close(server_sock);
    return 0;
}