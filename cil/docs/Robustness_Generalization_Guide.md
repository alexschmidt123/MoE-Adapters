# How to Make Your Model Better (Simple Version)

You have a model that learns new things step by step. Here are 3 simple ways to make it better and to know if it really works.

---

## 1. Try Different "Orders" (Like Shuffling Cards)

**What is the problem?**  
Your model learns the classes in one fixed order (first these 11, then these 10, then these 10…). Maybe it is good only because of that one order. We don’t know.

**What to do:**  
Think of the 101 classes like 101 cards. Right now you always show them in the same order.  
**Shuffle the cards** (change the order) a few times. Run your experiment each time with a new order.  
If the model still does well every time → it is **strong**. If it does well only with one order → it is **weak**.

**How (step by step):**
1. From the `cil` folder, run this once:
   ```
   python scripts/generate_class_orders.py --dataset food101 --num_classes 101 --seeds 42 123 456 --out_dir class_orders
   ```
   This makes 3 different “orders” of the 101 classes and saves them.

2. Run your experiment 3 times. Each time, use a different order:
   - First time: use `class_orders/food101_seed42.yaml`
   - Second time: use `class_orders/food101_seed123.yaml`
   - Third time: use `class_orders/food101_seed456.yaml`

3. Write down the final accuracy number for each run. If the 3 numbers are close to each other, your model is robust (strong). If one number is much lower, the model depends too much on the order.

---

## 2. Try More Than One Type of Data

**What is the problem?**  
Your model might be good only on food photos (Food-101). We don’t know if it is good on other things (e.g. animals, objects).

**What to do:**  
Use the **same** model and the **same** way of training, but run it on **different** datasets:
- Run once on CIFAR-100 (small images, 100 classes).
- Run once on Food-101 (food photos, 101 classes).
- Run once on TinyImageNet (images, 200 classes) if you have it.

Write down the result (final accuracy) for each.  
If the model is good on **all** of them → it **generalizes** (works on more types of data). If it is good only on one → it only learned that one type.

**How (step by step):**
1. Run:
   ```
   bash run.sh configs/class/cifar100_2-2-MoE-Adapters-N4-GoE.yaml
   ```
   Write down the final accuracy.

2. Run:
   ```
   bash run.sh configs/class/food101_11-10-MoE-Adapters-N4-GoE.yaml
   ```
   Write down the final accuracy.

3. If you have TinyImageNet, run that too and write down the number.

4. Put the 3 numbers in a small table. If all are high, your model generalizes well.

---

## 3. Try "Easy" and "Hard" Settings

**What is the problem?**  
Maybe the model is good only when the task is easy (few steps, many classes per step). We want to see if it is still good when the task is hard (many steps, few classes per step).

**What to do:**  
- **Easy:** Use 20 classes in the first task, 20 in the second, etc. (fewer tasks, each task has more classes).
- **Hard:** Use 5 classes per task (more tasks, each task has fewer classes).

Run your experiment once in the easy way and once in the hard way. Write down the accuracy for both.  
If the model is still good in the hard way → it is **robust**. If it drops a lot when you make it hard → it is not so robust.

---

## One-Page Summary (What to Do)

| What you want to check | What you do |
|------------------------|------------|
| Is the model strong no matter the order of classes? | Run the same experiment 3 times with 3 different class orders (use the script to create them). Write down the 3 accuracy numbers. If they are similar, the model is robust. |
| Does the model work on different types of data? | Run the same method on CIFAR-100, then on Food-101, then on TinyImageNet. Write down the accuracy for each. If all are good, the model generalizes. |
| Does the model work when the task is hard? | Run once with “easy” setting (e.g. 20 classes per task) and once with “hard” (e.g. 5 per task). Compare the two accuracy numbers. |

---

## The Script (Create Different Orders)

From the `cil` folder, run:

```
python scripts/generate_class_orders.py --dataset food101 --num_classes 101 --seeds 42 123 456 --out_dir class_orders
```

This creates 3 files. Each file is a different order of the 101 classes. Then you run your experiment 3 times: each time you tell the program to use a different file (seed42, then seed123, then seed456). At the end you have 3 accuracy numbers. If they are close, the model is strong.
