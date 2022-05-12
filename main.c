// Make conflict
#include <stdio.h>

#include "myheader.h"
int main(void) {
char name[17] = {0,};
printf("Name: ");
scanf("%s", name);
printHello();
printBye();
// 수정
// 수정2
return 0;
}
