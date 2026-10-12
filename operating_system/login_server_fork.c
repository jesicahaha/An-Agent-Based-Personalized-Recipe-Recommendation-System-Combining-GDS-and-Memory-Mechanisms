#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>//POSIX API（read, write, close, fork, sleep)
#include <arpa/inet.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/wait.h>

#define PORT 12345 //伺服器監聽的端口
#define BUF_SIZE 1024 //緩衝區大小
#define USER_FILE "users.txt" //用戶名和密碼文件

//檢查使用者名稱與密碼是否正確
int check_credentials(const char *username, const char *password) {
    FILE *fp = fopen(USER_FILE, "r");//用唯獨打開 沒有此檔案會回傳NULL
    if (!fp) return 0;
    char u[128], p[128];
    int ok = 0;//=0表示未成功
    while(fscanf(fp, "%s %s", u, p) != EOF) {//以空白分隔方式讀取兩個字串
        if(strcmp(u, username) == 0 && strcmp(p, password) == 0) {
            ok = 1;
            break;
        }
    }
    fclose(fp);
    return ok;
}

//處理一位客戶端的請求
void handle_client(int client_sock) {
    char buf[BUF_SIZE];
    //從client_sock裡讀資料 最多BUF_SIZE-1 存到buf 並回傳byte數
    int n = read(client_sock, buf, BUF_SIZE-1);
    if(n > 0) {
        buf[n] = '\0';//read()不會自動加\0
        char username[128], password[128];
        sscanf(buf, "%s %s", username, password);
        if(check_credentials(username, password))
            write(client_sock, "OK", 2);
        else
            write(client_sock, "FAIL", 4);
    }
    sleep(5);
    close(client_sock);
}


int main() {
    int server_sock, client_sock;
    struct sockaddr_in server_addr, client_addr;//sockaddr_in是IPv4的socket地址格式
    socklen_t addr_len = sizeof(client_addr);

    server_sock = socket(AF_INET, SOCK_STREAM, 0);//建立一個 IPv4、TCP的socket 若失敗會回傳 -1
    server_addr.sin_family = AF_INET;//給需要用到這個結構的function知道是用IPv4
    server_addr.sin_port = htons(PORT);//主機位元序轉為網路位元序
    server_addr.sin_addr.s_addr = INADDR_ANY;//接受任何IP位址的連線

    //把server_sock socket綁定到特定的IP與Port
    bind(server_sock, (struct sockaddr*)&server_addr, sizeof(server_addr));
    listen(server_sock, 10);//最多可以同時排隊 10 個連線

    printf("Fork server listening on port %d...\n", PORT);

    while(1) {//不斷接受新連線
        //阻塞等待新連線到來 並且建立一個新的socket來和這個客戶端通訊
        client_sock = accept(server_sock, (struct sockaddr*)&client_addr, &addr_len);
        if(client_sock < 0) continue;
        pid_t pid = fork();//建立子程序 父程序會得到子pid 子程序得到0
        if(pid == 0) {
            close(server_sock);
            handle_client(client_sock);
            exit(0);
        } else {
            close(client_sock);
            waitpid(-1, NULL, WNOHANG);//用來避免殭屍程序
        }
    }
    close(server_sock);
    return 0;
}