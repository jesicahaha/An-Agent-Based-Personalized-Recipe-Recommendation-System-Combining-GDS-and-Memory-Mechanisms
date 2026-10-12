#include <stdio.h> 
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>
//客戶端
#define PORT 12345
#define BUF_SIZE 1024

int main() {
    int sock;//socket描述符
    struct sockaddr_in server_addr;//宣告IPv4地址結構 用於指定伺服器IP與port。
    char username[128], password[128], buf[BUF_SIZE];

    printf("Username: ");
    scanf("%s", username);
    printf("Password: ");
    scanf("%s", password);

    sock = socket(AF_INET, SOCK_STREAM, 0);
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(PORT);
    server_addr.sin_addr.s_addr = inet_addr("127.0.0.1");//localhost

    if(connect(sock, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
        //connect()成功回傳0 失敗回傳-1
        perror("connect failed");
        return 1;
    }

    sprintf(buf, "%s %s", username, password);
    write(sock, buf, strlen(buf));//將帳密字串發送給伺服器(然後他程式碼的read接收)
    int n = read(sock, buf, BUF_SIZE-1);//讀取伺服器回傳資料
    if(n > 0) {
        buf[n] = '\0';
        printf("Server response: %s\n", buf);
    }

    close(sock);
    return 0;
}