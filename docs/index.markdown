---
layout: page
title: Baseball Pitch
---

## Project Description

Aim to perform real-time Baseball Pitch Classification (what type of pitch is being thrown). 

Theoretically what steps would one take to be able to do this?
- Collect a lot of videos of Baseball Pitches
- Label these videos based on Pitch Type
- Train a classifier of some kind
- Profit!

It doesn't end up this simple, but this is the general layout.

### Scope Limits

Initially I was going to attempt this problem with as many baseball videos as possible; more data is more better, right?
Turns out there are a lot of baseball games and even more baseball videos. It ends up as several hundreds of **terabytes** of videos. 

I don't have that much storage, so we have to set some limits for ourselves:

**Trim videos to the three seconds where the pitch happens.** For a large portion of these videos the pitcher is just standing there, or it cuts to the outfield after the ball is already hit. If we trim the first two seconds and only keep three seconds after that point, we get the part of the video where the windup and throw happen, cutting down on the amount of video we need to store, or label (and the more informative the frames are for our classifier).

**Just one pitcher: Bryan Woo.** This is the main limit, and it gives a lot of benefits:
- Cuts the 18 different types of baseball pitches down to **5 types** (Fastball, Sinker, Changeup, Slider, Sweeper) since these are the only types Bryan throws.
- No need to figure out left hand pitching. Bryan Woo is a **Right Hand Pitcher (RHP)** so we don't have to worry about mirroring videos or other ways to tackle handedness in classifying.
- Less overall videos. Bryan Woo is the same age as me - that is to say young -  so there are less videos of him pitching in total. That's less videos to label! It ends up as 28 GB untrimmed, **6.5 GB of videos when trimmed** (Still ends up as ~100 for storing the frame labels but I'm confident in ways to just store the videos and one text file of labels per video and have it break down frame by frame later in the future. Sadly it's messy to figure out how to do things the optimal way).
- He's also my brother's favorite pitcher, and a discussion of his interesting pitch style is what initially inspired this project. He has a similar arm placement for both **his Fastball and Sinker making them hard to tell apart**.


## Webscrapping

The MLB has an official website where they post baseball stats and videos of all the pitches from their games: [Baseball Savant](https://baseballsavant.mlb.com/). 

## Video Labeling

It would take an incredible amount of time to label all these videos even after trimming. 
