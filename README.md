## Amazon Frequetly asked
taken from https://leetcode.com/discuss/interview-question/488887/Amazon-Final-Interview-Questions-or-SDE1


[1. Two Sum](#1-two-sum)

[5. Longest Palindromic Substring](#5-longest-palindromic-substring)

[8. String to Integer (atoi)](#8-string-to-integer-atoi)

[12. Integer to Roman](#12-integer-to-roman)

[13. Roman to Integer](#13-roman-to-integer)

[20. Valid Parentheses](#20-valid-parentheses)

[23. Merge k Sorted Lists](#23-merge-k-sorted-lists)

[36. Valid Sudoku](#36-valid-sudoku)

[39. Combination Sum](#39-combination-sum)

[46. Permutations](#46-permutations)

[56. Merge Intervals](#56-merge-intervals)

[61. Rotate List](#61-rotate-list)

[64. Minimum Path Sum](#64-minimum-path-sum)

[79. Word Search](#79-word-search)

[98. Validate Binary Search Tree](#98-validate-binary-search-tree)

[100. Same Tree](#100-same-tree)

[101. Symmetric Tree](#101-symmetric-tree)

[102. Binary Tree Level Order Traversal](#102-binary-tree-level-order-traversal)

[109. Convert Sorted List to Binary Search Tree](#109-convert-sorted-list-to-binary-search-tree)

[116. Populating Next Right Pointers in Each Node](#116-populating-next-right-pointers-in-each-node)

[121. Best Time to Buy and Sell Stock](#121-best-time-to-buy-and-sell-stock)

[127. Word Ladder](#127-word-ladder)

[155. Min Stack](#155-min-stack)

[200. Number of Islands](#200-number-of-islands)

[207. Course Schedule](#207-course-schedule)

### 1. Two Sum
```java
class Solution {
    public int[] twoSum(int[] nums, int target) {
        HashMap<Integer, Integer> hm = new HashMap<>();
        for(int i=0; i< nums.length; ++i) {
            if(hm.containsKey(target-nums[i])) {
                return new int[] {hm.get(target-nums[i]), i};
            }
            hm.put(nums[i], i);
        }
        return null;
    }
}
```
### 5. Longest Palindromic Substring
   
```java
class Solution {
    public String longestPalindrome(String s) {
        if(s == null || s.length() < 1) return "";
        int start = 0, end =0;
        for(int i=0; i< s.length(); ++i){
            int l1 = expandAroundCenter(s, i, i);
            int l2 = expandAroundCenter(s, i, i+1);
            int max_len = Math.max(l1, l2);
            if(max_len > end - start){
                start = i - (max_len - 1) /2 ;
                end = i + (max_len) /2;
            }
        }
        return s.substring(start, end + 1);
    }
    public int expandAroundCenter(String s, int left, int right){
        int l = left, r = right;
        while(l >=0 && r < s.length() && s.charAt(l) == s.charAt(r))
        {
            --l;
            ++r;
        }
        return r - l - 1;
    }
}
```

### 8. String to Integer (atoi)

```java
class Solution {
    public int myAtoi(String str) {
        str = str.trim();
        if(str.length() == 0) return 0;
        String signs = "+-";
        if(!Character.isDigit(str.charAt(0)) && signs.indexOf(str.charAt(0)) == -1){
            return 0;
        }
        int sign = 1, start = 0;
        if(signs.indexOf(str.charAt(0)) != -1){
            start = 1;
            if(str.charAt(0) == '-') sign = -1;
        }
        Long num = 0L;
        for(int i=start; i<str.length(); ++i){
            if(!Character.isDigit(str.charAt(i))) break;
            num = num * 10L + Character.getNumericValue(str.charAt(i)) * 1L;
            if(num * sign > Integer.MAX_VALUE) return Integer.MAX_VALUE;
            if(num * sign < Integer.MIN_VALUE) return Integer.MIN_VALUE;
        }
        System.out.println(num);
        return (int) (long) num * sign;
    }
}
```

### 12. Integer to Roman
12 - solution from leetcode

```java
class Solution {
    int[] values = {1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1};    
    String[] symbols = {"M","CM","D","CD","C","XC","L","XL","X","IX","V","IV","I"};

    public String intToRoman(int num) {
        StringBuilder sb = new StringBuilder();
        // Loop through each symbol, stopping if num becomes 0.
        for (int i = 0; i < values.length && num >= 0; i++) {
            // Repeat while the current symbol still fits into num.
            while (values[i] <= num) {
                num -= values[i];
                sb.append(symbols[i]);
            }
        }
        return sb.toString();
    }
}
```

### 13. Roman to Integer

```java
class Solution {
    public int romanToInt(String s) {
        if(s.length() == 0) return 0;
        HashMap<Character, Integer> romans = new HashMap<>();
        romans.put('I', 1);
        romans.put('V', 5);
        romans.put('X', 10);
        romans.put('L', 50);
        romans.put('C', 100);
        romans.put('D', 500);
        romans.put('M', 1000);
        int num = romans.get(s.charAt(s.length() - 1));
        for(int i= s.length()-2; i>=0 ; --i){
            if(romans.get(s.charAt(i)) < romans.get(s.charAt(i+1))) 
                num = num - romans.get(s.charAt(i));
            else num = num + romans.get(s.charAt(i));
        }
        return num;
    }
}
```


### 20. Valid Parentheses
```java
class Solution {
    public boolean isValid(String s) {
        HashMap<Character, Character> braces = new HashMap<>();
        braces.put('(', ')');
        braces.put('{', '}');
        braces.put('[', ']');
        Stack<Character> validbrace = new Stack<>();
        for(char c: s.toCharArray()){
            if(braces.containsKey(c)) validbrace.push(c);
            else {
                if(validbrace.size() == 0) return false;
                char temp = validbrace.pop();
                if(braces.get(temp) != c) return false;
            }
        }
        if(validbrace.size() == 0) return true;
        return false;
    }
}
```

### 23. Merge k Sorted Lists

```java
class Solution {
    public ListNode mergeKLists(ListNode[] lists) {
        PriorityQueue<ListNode> pq = new PriorityQueue<ListNode>((l1, l2) -> l1.val - l2.val);
        for(ListNode node : lists){
            while(node != null){
                pq.add(node);
                node = node.next;
            }
        }
        ListNode head= new ListNode(0);
        ListNode curr = head;
        while(!pq.isEmpty()){
            ListNode temp = pq.poll();
            curr.next = temp;
            curr = temp;
        }
        curr.next = null;
        return head.next;
    }
}
```

### 36. Valid Sudoku
```java
class Solution {
    public boolean isValidSudoku(char[][] board) {
        HashMap<Integer, Set<Character>> rows = new HashMap<>();
        HashMap<Integer, Set<Character>> cols = new HashMap<>();
        HashMap<Integer, Set<Character>> boxes = new HashMap<>();
        for(int i=0; i< board.length; ++i){
            for(int j=0; j< board[0].length; ++j){
                if(board[i][j] != '.'){
                    Set<Character> row = rows.getOrDefault(i, new HashSet<Character>());
                    if(row.contains(board[i][j])) return false;
                    else row.add(board[i][j]);
                    rows.put(i, row);
                    Set<Character> col = cols.getOrDefault(j, new HashSet<Character>());
                    if(col.contains(board[i][j])) return false;
                    else col.add(board[i][j]);
                    cols.put(j, col);
                    Set<Character> box = boxes.getOrDefault(((i / 3 ) * 3 + j / 3), new HashSet<Character>());
                    if(box.contains(board[i][j])) return false;
                    else box.add(board[i][j]);
                    boxes.put((i / 3 ) * 3 + j / 3, box);
                }
            }
        }
        return true;
    }
}
```

### 39. Combination Sum

```java
class Solution {
    public void backtrack(List<List<Integer>> result, List<Integer> temp, int target, int[] candidates, int start){
        if(target < 0) return;
        if(target == 0) {
            result.add(new ArrayList<>(temp));
            return;
        }
        for(int i=start; i< candidates.length; ++i){
            temp.add(candidates[i]);
            backtrack(result, temp, target - candidates[i], candidates, i);
            temp.remove(temp.size() - 1);
        }
        return;
    }
    public List<List<Integer>> combinationSum(int[] candidates, int target) {
        List<List<Integer>> result = new ArrayList<>();
        List<Integer> temp = new ArrayList<>();
        backtrack(result, temp, target, candidates, 0);
        return result;
    }
}
```

### 46. Permutations
```java
class Solution {
  public void backtrack(int n,
                        ArrayList<Integer> nums,
                        List<List<Integer>> output,
                        int first) {
    if (first == n)
      output.add(new ArrayList<Integer>(nums));
    for (int i = first; i < n; i++) {
      Collections.swap(nums, first, i);
      backtrack(n, nums, output, first + 1);
      Collections.swap(nums, first, i);
    }
  }

  public List<List<Integer>> permute(int[] nums) {
    List<List<Integer>> output = new LinkedList();
    ArrayList<Integer> nums_lst = new ArrayList<Integer>();
    for (int num : nums)
      nums_lst.add(num);
    int n = nums.length;
    backtrack(n, nums_lst, output, 0);
    return output;
  }
}
```

### 56. Merge Intervals

```java
class Solution {
    public int[][] merge(int[][] intervals) {
        Arrays.sort(intervals, (a, b) -> a[0] - b[0]);
        LinkedList<int[]> merged = new LinkedList<>();
        for(int[] interval: intervals){
            if(merged.isEmpty() || merged.getLast()[1] < interval[0])
                merged.add(interval);
            else {
                merged.getLast()[1] = Math.max(merged.getLast()[1], interval[1]);
            }
        }
        return merged.toArray(new int[merged.size()][]);
    }
}
```

### 61. Rotate List
```java
class Solution {
    public ListNode rotateRight(ListNode head, int k) {
        if(head == null) return null;
        int n=0;
        ListNode temp = head;
        while(temp.next != null){
            ++n;
            temp = temp.next;
        }
        temp.next = head;
        ++n;
        temp = head;
        if(k >= n) k = k % n;
        for(int i=0; i< n-k-1; ++i){
            temp = temp.next;
        }
        head = temp.next;
        temp.next = null;
        return head;
    }
}
```

### 64. Minimum Path Sum
```java
public class Solution {
    public int minPathSum(int[][] grid) {
        int[][] dp = new int[grid.length][grid[0].length];
        for (int i = grid.length - 1; i >= 0; i--) {
            for (int j = grid[0].length - 1; j >= 0; j--) {
                if(i == grid.length - 1 && j != grid[0].length - 1)
                    dp[i][j] = grid[i][j] +  dp[i][j + 1];
                else if(j == grid[0].length - 1 && i != grid.length - 1)
                    dp[i][j] = grid[i][j] + dp[i + 1][j];
                else if(j != grid[0].length - 1 && i != grid.length - 1)
                    dp[i][j] = grid[i][j] + Math.min(dp[i + 1][j], dp[i][j + 1]);
                else
                    dp[i][j] = grid[i][j];
            }
        }
        return dp[0][0];
    }
}
```

### 79. Word Search
```java
class Solution {
    public boolean exist(char[][] board, String word) {
        for(int i = 0; i<board.length; ++i){
            for(int j=0; j < board[0].length; ++j){
                boolean temp = backtrack(board, word, i, j, 0);
                if(temp == true) return true; 
            }
        }
        return false;
    }
    public boolean backtrack(char[][] board, String word, int i, int j, int index){
        if(index == word.length()) return true;
        if(i<0 || j<0 || i >= board.length || j >= board[0].length || board[i][j] != word.charAt(index))
            return false;
        char temp = board[i][j];
        board[i][j] = '#';
        int[][] dirs = {{0,1},{1,0},{0,-1},{-1,0}};
        boolean ret = false;
        for(int[] dir: dirs){
            ret = backtrack(board, word, i + dir[0], j + dir[1], index+1);
            if(ret == true) break;
        }
        board[i][j] = temp;
        return ret;
    }
}
```

### 98. Validate Binary Search Tree
```java
class Solution {
    public boolean helper(TreeNode root, Integer lower, Integer upper){
        if(root == null) return true;
        if(lower != null && root.val <= lower) return false;
        if(upper != null && root.val >= upper) return false;
        if(!helper(root.left, lower, root.val)) return false;
        if(!helper(root.right, root.val, upper)) return false;
        return true;
    }
    public boolean isValidBST(TreeNode root) {
        return helper(root, null, null);
    }
}
```

### 100. Same Tree
```java
class Solution {
    public boolean isSameTree(TreeNode p, TreeNode q) {
        if(p == null && q == null) return true;
        if(p == null || q == null) return false;
        if(p.val != q.val) return false;
        return isSameTree(p.left, q.left) && isSameTree(p.right, q.right);
    }
}
```

### 101. Symmetric Tree
```java
class Solution {
    public boolean helper(TreeNode p, TreeNode q){
        if(p == null && q == null) return true;
        if(p == null || q == null) return false;
        if(p.val != q.val) return false;
        return helper(p.left, q.right) && helper(p.right, q.left);
    }
    public boolean isSymmetric(TreeNode root) {
        return helper(root, root);
    }
}
```

### 102. Binary Tree Level Order Traversal
```java
class Solution {
    public List<List<Integer>> levelOrder(TreeNode root) {
        Queue<TreeNode> q = new LinkedList<>();
        List<List<Integer>> result = new ArrayList<>();
        if(root != null) q.add(root);
        while(!q.isEmpty()){
            int len = q.size();
            List<Integer> temp = new ArrayList<>();
            for(int i=0; i< len; ++i){
                TreeNode node = q.remove();
                temp.add(node.val);
                if(node.left != null) q.add(node.left);
                if(node.right != null) q.add(node.right);
            }
            result.add(temp);
        }
        return result;
    }
}
```

### 109. Convert Sorted List to Binary Search Tree
```java
class Solution {
    public TreeNode helper(List<ListNode> nodes, int start, int end){
        int mid = (start + end) / 2;
        if(end < start) return null;
        TreeNode head = new TreeNode(nodes.get(mid).val);
        if(end == start) return head;
        head.left = helper(nodes, start, mid - 1);
        head.right = helper(nodes, mid + 1, end);
        return head;
    }
    public TreeNode sortedListToBST(ListNode head) {
        List<ListNode> nodes = new ArrayList<>();
        while(head != null){
            nodes.add(head);
            head = head.next;
        }
        return helper(nodes, 0, nodes.size() - 1);
    }
}
```


### 116. Populating Next Right Pointers in Each Node
Time complexity: O(N)
Space complexity: O(N)
```java
class Solution {
    public Node connect(Node root) {
        Queue<Node> q = new LinkedList<>();
        if(root == null) return root;
        q.add(root);
        while(!q.isEmpty()) {
            int len = q.size();
            for(int i=0; i<len; ++i){
                Node temp = q.remove();
                if(i < len-1) temp.next = q.peek();
                if(temp.left != null) q.add(temp.left);
                if(temp.right != null) q.add(temp.right);
            }
        }
        return root;
    }
}
```

### 121. Best Time to Buy and Sell Stock
time - O(n)
space - O(1)
```java
class Solution {
    public int maxProfit(int[] prices) {
        if(prices.length == 0) return 0;
        int maxProfit =0, min = prices[0];
        for(int i: prices){
            if(i < min) min = i;
            else maxProfit = Math.max(maxProfit, i - min);
        }
        return maxProfit;
    }
}
```

### 127. Word Ladder
Time - O(M^2 * N)
Space - O(M^2 * N)
```java
class Solution {
    public int ladderLength(String beginWord, String endWord, List<String> wordList) {
        HashMap<String, List<String>> dic = new HashMap<>();
        int word_len = wordList.get(0).length();
        for(int i=0; i < wordList.size(); ++i){
            for(int j=0; j< word_len; ++j){
                String word= wordList.get(i);
                String new_word = word.substring(0, j) + '*' + word.substring(j + 1, word_len);
                List<String> word_list = dic.getOrDefault(new_word, new ArrayList<String>());
                word_list.add(word);
                dic.put(new_word, word_list);
            }
        }
        // BFS of graph
        int level = 1;
        Queue<String> q = new LinkedList<>();
        HashSet<String> visited = new HashSet<>();
        q.add(beginWord);
        while(!q.isEmpty()){
            String word = q.poll();
            for(int i=0; i< word_len; ++i){
                String new_word = word.substring(0, i) + '*' + word.substring(i + 1, word_len);
                List<String> word_list = dic.getOrDefault(new_word, new ArrayList<String>());
                for(String adjWord: word_list){
                    if(adjWord.equals(endWord)) return level ;
                    if(!visited.contains(adjWord)) {
                        visited.add(adjWord);
                        q.add(adjWord);
                    }
                }
            }
            ++level;
        }
        return 0;
    }
}
```

### 155. Min Stack
```java
class MinStack {

    /** initialize your data structure here. */
    List<Integer> minStack;
    List<Integer> minList;
    public MinStack() {
        minStack = new ArrayList<>();
        minList = new ArrayList<>();
    }
    
    public void push(int x) {
        minStack.add(x);
        if(minList.size() == 0) minList.add(x);
        else {
            int temp = minList.size() - 1;
            minList.add(Math.min(x, minList.get(temp)));
        }
    }
    
    public void pop() {
        if(minStack.size() >=0 ) {
            minStack.remove(minStack.size() - 1);
            minList.remove(minList.size() - 1);
        }
    }
    
    public int top() {
        return minStack.get(minStack.size() -1 );
    }
    
    public int getMin() {
        return minList.get(minList.size() - 1);
    }
}
```

### 200. Number of Islands
```java
class Solution {
    public void helper(char[][] grid, int i, int j){
        if(i < 0 || j < 0 || i >= grid.length || j >= grid[0].length || grid[i][j] != '1')
            return;
        grid[i][j] = '0';
        helper(grid, i+1, j);
        helper(grid, i, j+1);
        helper(grid, i-1, j);
        helper(grid, i, j-1);
        return;
    }
    public int numIslands(char[][] grid) {
        int count =0;
        for(int i=0; i<grid.length; ++i){
            for(int j=0; j<grid[0].length; ++j){
                if(grid[i][j] == '1'){
                    ++count;
                    helper(grid, i , j);
                }
            }
        }
        return count;
    }
}
```

### 207. Course Schedule
```java
class Solution {
    public boolean canFinish(int numCourses, int[][] prerequisites) {
        HashMap<Integer, List<Integer>> adj_list = new HashMap<>();
        for(int i=0; i<prerequisites.length; ++i){
            List<Integer> num_list = adj_list.getOrDefault(prerequisites[i][1],
                                                          new ArrayList<>());
            num_list.add(prerequisites[i][0]);
            adj_list.put(prerequisites[i][1], num_list);
        }
        boolean[] path = new boolean[numCourses];
        Arrays.fill(path, false);
        for(int i=0; i<numCourses; ++i){
            if(dfs(adj_list, path, i)) return false;
        }
        return true;
    }
    public boolean dfs(HashMap<Integer, List<Integer>> adj_list, boolean[] path, int curr_ind){
        if(path[curr_ind]) return true;
        if(!adj_list.containsKey(curr_ind)) return false;
        path[curr_ind] = true;
        boolean ret = false;
        for(Integer nextCourse: adj_list.get(curr_ind)){
            ret = dfs(adj_list, path, nextCourse);
            if(ret) break;
        }
        path[curr_ind] = false;
        return ret;
    }
}
```