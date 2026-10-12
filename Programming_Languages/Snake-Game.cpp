#include <iostream>
#include <ctime>
#include <conio.h>
#include <windows.h>

#define DIRECTION 224  
#define UP   72 
#define DOWN 80 
#define LEFT 75 
#define RIGHT 77

using namespace std;

void HideCursor();
void gotoxy(int,int);

void HideCursor(){
    CONSOLE_CURSOR_INFO cursor_info={1, 0};
    SetConsoleCursorInfo(GetStdHandle(STD_OUTPUT_HANDLE),&cursor_info);
}

void gotoxy(int x, int y) {
    COORD  loc;
    HANDLE hOuput=GetStdHandle(STD_OUTPUT_HANDLE); 
    loc.X=x;
    loc.Y=y;
    SetConsoleCursorPosition(hOuput,loc);
} 

class Object{
protected:
	static int WIDTH;
    static int HEIGHT;
    char shape;
public:
	Object():shape(' '){
	}
	
	Object(char skin):shape(skin){
	}
	
	virtual void Show()=0;
};

int Object::WIDTH=70;
int Object::HEIGHT=20;

class Map{
private:
	char map[70][20];
public:
	Map(){
		for(int i=0;i<70;i++){
			for(int k=0;k<20;k++){
				map[i][k]=' ';
			}
		}
	}
	
	void Input(int x,int y,char shape){
		map[x][y]=shape;
	}
	
	void Remove(int x,int y){
		map[x][y]=' ';
	}
};


class Snake:public Object{
private:
	Map snakeMap;
	int *x;
	int *y;
	int *temx;
	int *temy;
	int head;
	char direction;
	char alive;
public:
	Snake():Object('S'),direction('R'),alive('A'),head(9){
		x=new int[10];
        y=new int[10];
        temx=new int[10];
        temy=new int[10];
        
        //從9開始為頭 
        x[9]=1;
        y[9]=1;
        for(int i=0;i<=8;i++){
        	x[i]=0;
        	y[i]=0;
		}
		
		for(int i=0;i<=9;i++){
			temx[i]=0;
        	temy[i]=0;
		}
		
        gotoxy(x[9],y[9]);
        cout<<shape;
        snakeMap.Input(x[9],y[9],shape);
	}
    
    void Move(){
        if(kbhit()){
            unsigned char key=getch();
            if(key==DIRECTION){
                switch(getch()){
                    case UP:
                        direction='U';
                        break;
                    case DOWN:
                        direction='D';
                        break;
                    case LEFT:
                        direction='L';
                        break;
                    case RIGHT:
                        direction='R';
                        break;
                }
            }
        }
        Forward();
        Sleep(200);
    }
    
    void Forward(){
        gotoxy(x[9],y[9]);
        cout<<' ';
        snakeMap.Remove(x[9],y[9]);
        
        //儲存以前的位置
		for(int i=9;i>=0;i--){
		    temx[i]=x[i];
			temy[i]=y[i];	
		} 
		
        for(int i=9;i>=0;i--){
        	if(x[i]!=0 && y[i]!=0){
        		head=i;
			}
		}
		
		
		for(int i=9;i>=head;i--){
			if(x[i]!=0 && y[i]!=0 && x[i-1]!=0 && y[i-1]!=0){
				x[i]=x[i-1];
				y[i]=y[i-1];
			}
		}
		
        if(direction=='U'){
            y[head]--;
        }else if(direction=='D'){
            y[head]++;
        }else if(direction=='L'){
            x[head]--;
        }else if(direction=='R'){
           x[head]++;
        }
        
        gotoxy(x[head],y[head]);
        cout<<shape;
        snakeMap.Input(x[head],y[head],shape);
    }

    void Addsnake(){
    	head=head-1; 
        if(direction=='U'){
            x[head]=x[head+1];
            y[head]=y[head+1]-1;
        }else if(direction=='D'){
            x[head]=x[head+1];
            y[head]=y[head+1]+1;
        }

        if(direction=='L'){
            x[head]=x[head+1]-1;
            y[head]=y[head+1];
        }else if(direction=='R'){
            x[head]=x[head+1]+1;
            y[head]=y[head+1];
        }
        
        gotoxy(x[head],y[head]);
        cout<<shape;
        snakeMap.Input(x[head],y[head],shape);
    }
    
    int Call_x(int num){
        return x[num]; 
    }

    int Call_y(int num){
        return y[num]; 
    }
    
    int Call_head(){
        return head; 
    }
    
    //贏的條件 
	int Win(){
		if(head==0){
			return 1;
		}
		return 0;
    }
	
	//確定蛇沒有超出地圖 
	char Suresnake(){
		for(int i=9;i>=head;i--){
			if(x[i]==0 || x[i]==Object::WIDTH-1 || y[i]==0 || y[i]==Object::HEIGHT-1){
				alive='F';
			}
		}
		
		return (alive);
    }
  
    //清空蛇所在的位置 
    ~Snake(){
    	for(int i=9;i>=head;i--){
    		gotoxy(x[i],y[i]);
    		cout<<' ';
    		snakeMap.Remove(x[i],y[i]);
		}
    }

    //偵測蛇是否吃到自己 
    int Eatsnake(){
    	for(int i=9;i>head;i--){
    		if(x[head]==temx[i] && y[head]==temy[i]){
    			return 1;
			}
		}
		return 0;
    }
  
    //顯示蛇的長度跟蛇頭的位置 
    void Show() override{
        gotoxy(72,1);
        cout<<"length:"<<(10-head);
        
        for(int i=87;i<=100;i++){
    		gotoxy(i,2);
    		cout<<" ";
		}
        gotoxy(72,2);
        cout<<"Snake head(x,y):("<<x[head]<<","<<y[head]<<")";
    }
};

class Egg:public Object{
private:
	Map eggMap;
    int egg_x;
    int egg_y;
public:
	Egg(Snake &snake):Object('E'){
        int flag=0;
        do{
            flag=0;
            do{
                egg_x=rand()%((Object::WIDTH)-1); // 1~19 
                egg_y=rand()%((Object::HEIGHT)-1); // 1~69 
            }while(egg_x==0 || egg_y==0 || egg_x==Object::WIDTH-1 || egg_y==Object::HEIGHT-1);

            for(int i=9;i>=snake.Call_head();i--){
                if(egg_x==snake.Call_x(i) && egg_y==snake.Call_y(i)){
                    flag=1;
                }
            }
        } while(flag==1);

        gotoxy(egg_x,egg_y);
        cout<<shape;
        eggMap.Input(egg_x,egg_y,shape);
        Show(); 
    }

    int Examine(Snake &snake){
        if(egg_x==snake.Call_x(snake.Call_head()) && egg_y==snake.Call_y(snake.Call_head())){
                return 1;
        }
        return 0;
    }
    
    int Call_Eggx(){
        return egg_x; 
    }

    int Call_Eggy(){
        return egg_y; 
    }
    
    //顯示蛋的位置 
    void Show() override{
    	for(int i=81;i<=100;i++){
    		gotoxy(i,3);
    		cout<<" ";
		}
		
        gotoxy(72,3);
        cout<<"Egg(x,y):("<<egg_x<<","<<egg_y<<")";
    }
    
    ~Egg(){
    	eggMap.Remove(egg_x,egg_y);
	}
};

class Wall:public Object{
private:
    Map wallMap;
public:
    Wall():Object('*'){
        Show();
    }
    
    void Show() override{
    	for(int i=0;i<=Object::WIDTH-1;i++){
            for(int k=0;k<=Object::HEIGHT-1;k++){
                if(i==0 || i==Object::WIDTH-1 || k==0 || k==Object::HEIGHT-1){
                    gotoxy(i,k);
                    cout<<shape;
                    wallMap.Input(i,k,shape);
                }
            }
        }
	}
};

class GM:public Object{ 
private:
    Wall *wall; 
    Snake *snake;
    Egg *egg;
    int losenumber;
    int winnumber;
public:
    GM():wall(new Wall),snake(new Snake),egg(new Egg(*snake)),losenumber(0),winnumber(0){}
    
    void Show() override{
    	gotoxy(72,4);
        cout<< "Number of Lose:"<<losenumber;
        gotoxy(72,5);
        cout<< "Number of Win:"<<winnumber;
	}

    void Run(){
        while(1) {
            snake->Move();

            if(egg->Examine(*snake)==1){
                snake->Addsnake();
                
                delete egg;
                Egg *egg=new Egg(*snake);
            }

            if((snake->Win())==1){
                gotoxy(1,1);
                cout<<"You Win! New Game Start!";
                Sleep(1000);
                for(int k=0;k<25;k++){
                    gotoxy(k,1);
                    cout<<' ';
                }

                winnumber++;
                delete snake;
                Snake *snake=new Snake;
                delete wall;
                Wall *wall=new Wall;
            }

            if(snake->Eatsnake()==1 || snake->Suresnake()=='F'){
                gotoxy(1,1);
                cout<<"Game over";
                Sleep(1000);
                for(int k=0;k<10;k++){
                    gotoxy(k,1);
                    cout<<' ';
                }

                losenumber++;
				delete snake;
                Snake *snake=new Snake;
                delete wall;
                Wall *wall=new Wall;
            }
            
            Show(); 
            snake->Show();
            egg->Show();
        }
    }
};

int main() {
    srand(time(NULL));
    HideCursor();
    
    GM *gm=new GM;
    gm->Run();
 
    delete gm;
 
    return 0;
}