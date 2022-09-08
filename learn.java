package com.jetbrains;

import java.util.*;

public class learn {
    static int fibonacci2(int number) {
        if(number == 1 || number == 2) {
            return 1;
        }
        return fibonacci2(number - 1) + fibonacci2(number + 1);
    }
    public static void main(String[] args) {
        int a = 1;
        int b = 1;
        int fibonacci = 1;
        int number = 10;

        for (int i = 3; i <= number; i++) {
            fibonacci = a + b;
            System.out.println("a" +a);
            a = b;
            System.out.println("b" + b);
            b = fibonacci;
            System.out.println(fibonacci + " ");
        }
    }

}


// charAt()