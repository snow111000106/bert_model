from sympy.codegen.ast import integer


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def create_linked_list(vals):
    dummy = ListNode()
    current = dummy
    for val in vals:
        current.next = ListNode(val)
        current = current.next
    return dummy.next


def print_linked_list(head):
    current = head
    vals = []
    while current:  # 遍历链表直到末尾
        vals.append(current.val)  # 将当前节点的值加入列表
        current = current.next       # 移动到下一个节点
    print(" -> ".join(map(str, vals)))


def my_code_1(a, target):
    """两数之和"""
    num = len(a)
    for i in range(num):
        for j in range(num-i-1):
            if a[i] + a[i+j+1] == target:
                return [i, i+j+1]


def my_code_9(x):
    """回文数"""
    if x < 0:
        return False
    else:
        num = list(map(int, str(x)))
        l = len(num)
        y = []
        for i in range(l):
            y.append(str(num[l - 1 - i]))
        s = "".join(y)
        if int(s) == x:
            return True
        else:
            return False


def my_code_13(num):
    """罗马数字转整数"""
    num_map = {
        'M': 1000,
        'D': 500,
        'C': 100,
        'L': 50,
        'X': 10,
        'V': 5,
        'I': 1
    }
    re = []
    for i, v in num_map.items():
        n = int(num/v)
        num = num - n*v
        for j in range(n):
            re.append(i)
    result = ''.join(re)
    return result


def my_code_14(sorts):
    """最长公共前缀"""
    min = len(sorts[0])
    for i in sorts:
        l = len(i)
        if l < min:
            min = l
    ww = ''
    for j in range(min+1):
        num = sorts[0][:j]
        count = 0
        for word in sorts:
            if word.startswith(num):
                count += 1
        if count >= len(sorts):
            ww = num
    return ww


def my_code_20_1(s):
    """有效的括号-硬解"""
    left = {
        '(': ')',
        '[': ']',
        '{': '}'
    }
    n = 0
    a = len(s)
    if a % 2 != 0 or s[0] in [']', '}', ')']:
        return False
    else:
        while s != '' and n < a/2:
            ss = list(s)
            l = len(s)
            for i in range(l-1):
                if left.get(ss[i]) == ss[i + 1]:
                    ss[i] = ''
                    ss[i+1] = ''
            s = ''.join(ss)
            n += 1
    return not s


def my_code_20_2(s):
    """有效的括号-栈解法"""
    left = {
        ')': '(',
        ']': '[',
        '}': '{'
    }
    stack = []
    for i in list(s):
        if stack and i in left.keys():
            if stack[-1] == left.get(i):
                stack.pop()
            else:
                stack.append(i)
        else:
            stack.append(i)
    return not stack


def my_code_21(l1,l2):
    """合并有序链表"""
    current_l1 = l1
    current_l2 = l2
    current = ListNode()
    head = current
    while current_l1 or current_l2:
        print_linked_list(current_l1)
        print_linked_list(current_l2)
        if not current_l1:
            current.next = current_l2
            break
        elif not current_l2:
            current.next = current_l1
            break
        elif current_l1.val <= current_l2.val:
            current.next = current_l1
            current_l1 = current_l1.next
        else:
            current.next = current_l2
            current_l2 = current_l2.next
        current = current.next
        print_linked_list(current)
    return head.next


def my_code_26(nums):
    """删除有序数组中的重复项"""
    i = 0
    l = len(nums)
    if l <=1:
        return l
    else:
        while i <= l-1:
            if nums[i] == nums[i+1]:
                del nums[i+1]
            else:
                i += 1
            if i == len(nums) - 1:
                break
    return len(nums),nums


def my_code_27(nums, val):
    """移除元素"""
    i = 0
    l = len(nums)
    while i <= l-1:
        if nums[i] == val:
            del nums[i]
        else:
            i += 1
        if i == len(nums):
            break
    return len(nums),nums\



def my_code_28(haystack, needle):
    """找出字符串中第一个匹配项的下标"""
    n = -1
    if len(needle)>len(haystack):
        return -1
    else:
        for i,v in enumerate(haystack):
            if v == needle[0]:
                if haystack[i:i+len(needle)] == needle:
                    return i
                else:
                    continue
        return n


def my_code_35(nums, target):
    """搜索插入位置"""
    for i,v in enumerate(nums):
        if v == target:
            return i
    if len(nums) == 1:
        if nums[0] < target:
            return 1
        else:
            return 0
    for i,v in enumerate(nums):
        if i >= len(nums)-1:
            return len(nums)
        else:
            if nums[i] < target < nums[i + 1]:
                return i + 1
            elif target < nums[i]:
                return 0


def my_code_58(s):
    """最后一个单词的长度"""
    n = 0
    for i in range(1, len(s)):
        if s[-i] == ' ':
            continue
        elif s[-i] != ' ' and s[-i-1] == ' ':
            break
        else:
            n += 1
    return n+1


def my_code_66(l):
    """加一"""
    for i in range(1, len(l)+1):
        if l[-i] == 9:
            l[-i] = 0
            if i == len(l) and l[-i] == 0:
                l.insert(0, 1)
                break
        else:
            l[-i] = l[-i] + 1
            break
    return l


def my_code_67(a, b):
    """二进制求和"""
    if len(a) < len(b):
        a = a.zfill(len(b))
    else:
        b = b.zfill(len(a))
    a = list(a)
    b = list(b)
    is_jw = 0
    for i in range(1, len(b)+1):
        c = int(b[-i])+int(a[-i])+is_jw
        if c >= 2:
            a[-i] = str(c % 2)
            if i == len(b):
                a[-i] = str(c % 2)
                a.insert(0, '1')
                break
            is_jw = 1
        else:
            a[-i] = str(c)
            is_jw = 0
    res = ''.join(a)
    return res


def my_code_69(x):
    """x的平方根"""
    n = 0
    while n*n <= x:
        n += 1
    return n-1


def my_code_69_2(x):
    """x的平方根-解法2"""
    l = 0
    r = x
    a = 0
    while l <= r:
        mid = (l+r)//2
        if mid * mid <= x:
            a = mid
            l = mid + 1
        else:
            r = mid - 1
    return a


def math_a(n):
    p = 1
    for i in range(1, n+1):
        p *= i
    return p


def math_comb(m, n):
    ans = math_a(m) // (math_a(m-n)*math_a(n))
    return ans


def my_code_70(x):
    """爬楼梯-组合解法"""
    i = 1
    count = 1
    while i <= x:
        if 2*i > x:
            break
        elif 2*i == x:
            count = count + 1
            break
        else:
            count = count+math_comb(x-i, i)
            i += 1
    return count


def my_code_70_2(n):
    """爬楼梯-斐波那契数列"""
    p = 0
    q = 1
    s = 1
    for i in range(1, n+1):
        s = p+q
        p = q
        q = s
    return s


def my_code_83(head):
    """删除排序链表中的重复值"""
    if not head:  # 如果链表为空，直接返回
        return None
    current = head
    while current.next:
        if current.val == current.next.val:
            current.next = current.next.next
        else:
            current = current.next

    return head


def my_code_88(nums1, m, nums2, n):
    """合并两个有序数组"""
    for i in range(n):
        nums1.insert(m+i, nums2[i])
        nums1.pop()
    nums1.sort()
    print(f'nums1:{nums1}')


def my_code_88_2(nums1, m, nums2, n):
    """合并两个有序数组-解法2"""
    p1 = m - 1  # nums1 中有效元素的最后一个索引
    p2 = n - 1  # nums2 中的最后一个索引
    p = m + n - 1  # nums1 最后位置的索引（合并后的最终长度）

    # 从后往前遍历，将较大的元素放在 nums1 的最后
    while p1 >= 0 and p2 >= 0:
        if nums1[p1] > nums2[p2]:
            nums1[p] = nums1[p1]
            p1 -= 1
        else:
            nums1[p] = nums2[p2]
            p2 -= 1
        p -= 1

    # 如果 nums2 中还有剩余元素，将其填入 nums1
    while p2 >= 0:
        nums1[p] = nums2[p2]
        p2 -= 1
        p -= 1

    print(f'nums1:{nums1}')


def my_code_100(p, q):
    """相同的树"""
    if p and q:
        if p.val != q.val:
            return False
        if my_code_100(p.right, q.right) and my_code_100(p.left, q.left):
            return True
        else:
            return False
    elif not p and not q:
        return True
    else:
        return False


def my_code_100_2(p, q):
    """相同的树-简化版本"""
    if not p and not q:
        return True
    if not p or not q or p.val != q.val:
        return False
    return my_code_100(p.left, q.left) and my_code_100(p.right, q.right)


def my_code_101(root):
    """对称二叉树"""
    if not root:
        return True

    def is_mirror(r,l):
        """左子树和右子树是否对称"""
        if not r and not l:
            return True
        if not r or not l:
            return False
        return r.val == l.val and is_mirror(r.right, l.left) and is_mirror(r.left, l.right)

    return is_mirror(root.right, root.left)


def my_code_104(root):
    """二叉树的最大深度"""
    if not root:
        return 0
    left_depth = my_code_104(root.left)
    right_depth = my_code_104(root.right)

    return max(left_depth, right_depth) + 1


def print_tree(root, level=0):
    """递归打印二叉树的结构"""
    if not root:
        return
    # 打印右子树
    print_tree(root.right, level + 1)
    # 打印当前节点
    print(" " * 4 * level + "->", root.val)
    # 打印左子树
    print_tree(root.left, level + 1)


def my_code_108(nums: list[int]) -> TreeNode:
    """将有序数组转换为平衡二叉搜索树"""

    def write_value(left, right):
        if left > right:
            return None
        mid = (right+left)//2
        root = TreeNode(nums[mid])

        root.left = write_value(left, mid-1)
        root.right = write_value(mid + 1, right)
        return root

    return write_value(0, len(nums)-1)


def my_code_110(root) -> bool:
    """判断是否是平衡二叉树-自下而上"""

    def check_balance(node: TreeNode):
        if not node:
            return 0  # 空树的高度是 0

        left_height = check_balance(node.left)
        if left_height == -1:  # 左子树不平衡
            return -1

        right_height = check_balance(node.right)
        if right_height == -1:  # 右子树不平衡
            return -1

        # 如果左右子树的高度差大于 1，返回 -1
        if abs(left_height - right_height) > 1:
            return -1

        # 返回当前树的高度
        return max(left_height, right_height) + 1

    # 如果返回值是 -1，说明树不平衡
    return check_balance(root) != -1


def my_code_111(root):
    """二叉树最小深度"""
    if not root:
        return 0
    if not root.left and not root.right:
        return 1
    if root.right and root.left:
        r = my_code_111(root.right)
        l = my_code_111(root.left)
        return min(l, r) + 1
    elif root.right:
        r = my_code_111(root.right)
        return r+1
    elif root.left:
        r = my_code_111(root.left)
        return r+1


def my_code_112(root, targetSum):
    """路径总和"""
    if not root:
        return False
    def sumRe(nood, a):
        if not nood:
            return False
        if not nood.right and not nood.left and a-nood.val == 0:
            return True
        return sumRe(nood.left, a-nood.val) or sumRe(nood.right, a-nood.val)

    return sumRe(root, targetSum)


def my_code_118(numRows):
    """杨辉三角"""

    n = 1
    res_list = []
    while n <= numRows:
        next_list = [1] * n
        if n >= 3:
            for i in range(1, n-1):
                next_list[i] = res_list[-1][i-1]+res_list[-1][i]

        res_list.append(next_list)
        n += 1
    return res_list


def my_code_119(rowIndex: int):
    """杨辉三角Ⅱ"""
    n = 1
    res_list = []
    while n <= rowIndex+1:
        next_list = [1] * n
        if n >= 3:
            for i in range(1, n - 1):
                next_list[i] = res_list[-1][i - 1] + res_list[-1][i]

        res_list.append(next_list)
        n += 1
    return res_list[rowIndex]


def my_code_121(prices):
    """买卖股票的最佳时机"""
    max_a = 0
    for i in range(len(prices)):
        v = max(prices[i:])
        if v > prices[i]:
            max_v = v - prices[i]
            if max_v > max_a:
                max_a = max_v
    return max_a


def my_code_121_2(prices):
    """买卖股票的最佳时机"""
    c = prices[0]
    p = 0
    for i in prices:
        c = min(c, i)
        p = max(p, i-c)
    return p


def my_code_125(s: str):
    """验证回文串"""
    import re
    s = s.lower()
    s_list = re.findall(r'[a-zA-Z0-9]', s)
    for i in range(1, len(s_list) // 2+1):
        if s_list[i-1] != s_list[-i]:
            return False
    return True


def my_code_136(nums: list[int]) -> int:
    """只出现一次的数字"""
    new = [nums[0]]
    for num in nums[1:]:
        if num in new:
            new.remove(num)
        else:
            new.append(num)
    return new[0]


def my_code_141(head: ListNode) -> bool:
    """环形链表"""
    seen = set()
    while head:
        if head in seen:
            return False
        seen.add(head)
        head = head.next
    return True


def my_code_144(root: TreeNode) -> list:
    """二叉树的前序遍历中-左-右"""
    new_list = []
    def test(node):
        if not node:
            return
        new_list.append(node.val)
        test(node.left)
        test(node.right)
    test(root)
    return new_list


def my_code_94(root):
    """二叉树的中序遍历左-中-右"""
    if not root:
        return []
    rest = []
    rest.extend(my_code_94(root.left))
    rest.append(root.val)
    rest.extend(my_code_94(root.right))
    return rest


def my_code_145(root: TreeNode) -> list:
    """二叉树的后序遍历,左-右-中"""
    new_list = []
    def test(node: TreeNode):
        if not node:
            return
        test(node.left)
        test(node.right)
        new_list.append(node.val)
    test(root)
    return new_list


def my_code_160_test(a:ListNode,b:ListNode):
    c = create_linked_list([8,4,5])
    aa = a
    bb = b

    while a:
        a = a.next
        if not a.next:
            a.next = c
            break
    while b:
        b = b.next
        if not b.next:
            b.next = c
            break
    return aa,bb


def my_code_160(headA:ListNode, headB: ListNode):
    """相交链表"""
    a,b = headA,headB
    while a and b:
        if a == b:
            return a
        a = a.next
        b = b.next
        if not a and not b:
            return
        if not a:
            a = headB
        if not b:
            b = headA


def my_code_168(columnNumber:int) -> str:
    """Excel表列名称"""
    new = []
    map_word = {i: chr(65 + i) for i in range(0, 26)}
    while columnNumber > 0:
        columnNumber -= 1
        new.insert(0, map_word[columnNumber%26])
        columnNumber //= 26
    return ''.join(new)


def my_code_169(nums: list[int]) -> int:
    """多数元素"""
    new_dict = {}
    for i in nums:
        k = new_dict.get(i)
        if k is None:
            new_dict.update({i: 1})
        else:
            new_dict.update({i: new_dict[i]+1})
    new_dict = sorted(new_dict.items(), reverse=True, key=lambda d:d[1])
    return new_dict[0][0]
    m,n =a,b
    while m and n:
        if m == n:
            return m
        m = m.next
        n = n.next
        if not m and not n:
            return
        if not m:
            m = b
        if not n:
            n = a


def my_code_171(columnTitle:str) -> int:
    """excel表列序号"""
    word_dict = {chr(i+64):i for i in range(1,27)}
    s = 0
    def counts(num, n):
        y = 1
        while n > 0:
            y *= 26
            n -= 1
        return y*num
    l = len(columnTitle)
    for i in columnTitle:
        s += counts(word_dict[i], l-1)
        l -= 1
    return s


def my_code_190(n:integer):
    """颠倒二进制位"""
    new = list(bin(n)[2:])
    y = ['0'] * (32 - len(new))
    if len(new) <= 32:
        y.extend(new)
    for i in range(1,17):
        x = y[i-1]
        y[i-1] = y[-i]
        y[-i] = x
    return int(''.join(y),2)


def my_code_191(n:int) -> int:
    """位1的个数"""
    new = list(bin(n)[2:])
    count = 0
    for i in new:
        if i == '1':
            count +=1
    return count

def my_code_202(n: int) -> bool:
    """快乐数"""
    new = set()
    while n !=1:
        sum = 0
        for i in list(str(n)):
            print(i)
            sum += int(i)**2
        if sum in new:
            return False
        new.add(sum)
        n = sum
    return True


def my_code_203(head:ListNode, val:int):
    """移除链表元素"""
    new = ListNode(0)
    current = new
    current.next = head
    a = current
    while a.next:
        if a.next.val == val:
            a.next = a.next.next
        else:
            a = a.next
    return current.next


def my_code_203_2(head:ListNode, val:int):
    """移除链表元素"""
    if not head:
        return None
    a = head
    while a.next:
        if a.next.val == val:
            a.next = a.next.next
        else:
            a = a.next
    if head.val == val:
        return head.next
    else:
        return head


def my_code_205(s :str,t:str) -> bool:
    """同构字符串"""
    pass

if __name__ == '__main__':
    # a = 8
    # r = my_code_202(a)
    # print(r)
    # a = create_linked_list([4, 1])
    b = create_linked_list([6,6,1,2])
    # ac,bc = my_code_160_test(a,b)
    r = my_code_203_2(b,6)
    print_linked_list(r)
    # bb = [-1,2,3]
    # mm, nn = 4, 3
    # root1 = TreeNode(1)
    # root1.right = TreeNode(2)
    # root1.right.right = TreeNode(4)
    # root1.right.right.right = TreeNode(5)
    # root1.right.right.left = TreeNode(4)
    # root1.right.right.right.right = TreeNode(6)
    # print_tree(root1)
    # print('结果')
    # r = my_code_145(root1)
    # print(r)

    # a=[2,2,2,'null',2,'null','null',2]
    # b=[2,2,2,2,'null',2,'null']
    #
    # r = my_code_100(root1, root2)
    # print(r)
    # # print_linked_list(re)
    # l1 = create_linked_list([0])
    # l2 = create_linked_list([0])
