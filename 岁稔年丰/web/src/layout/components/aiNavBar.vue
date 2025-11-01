<template>
  <div class="NavBar">
    <!-- <div class="ai-navbar">
          <span>睿农 <span>基于改进的ssd农作物病虫害识别系统</span> </span>
        </div>

      </div>

      <div class="nav-right">
        <div class="tag" @click="showConsole">数 据 中 心</div>
      </div>
    </div> -->

  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, watch } from 'vue';
import { useRouter } from 'vue-router'
import type { TabsPaneContext } from 'element-plus'

const router = useRouter();
const showConsole = () => {
  router.push("/algorithm01")
}

</script>


<style lang="scss" scoped>
.NavBar {
  width: 100%;
  position: fixed;
  box-shadow: 0 0 10px 0 rgba(0, 0, 0, 0.2);
  top: 0;
  z-index: 100;
  min-width: 1260px;
}

.ai-navbar {
  margin: 0 auto;
  min-height: 60px;
  box-sizing: border-box;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 20px 8%;
}


.cancelFixed {
  position: relative;
}

.nav-left {
  display: flex;
  align-items: center;
  position: relative;
}

.logo {
  height: 30px;
  margin-right: 40px;
  cursor: pointer;
  line-height: 30px;

  span {
    color: #FFBA01;
    font-size: 32px;

    span {
      color: #b9e50d;
      font-size: 24px;
    }
  }
}

.nav-right {
  font-size: 16px;
  font-weight: 500;
  color: #fff;
  display: flex;
  align-items: center;

  .tag {
    font-weight: bold;
    margin-right: 40px;
    cursor: pointer;
    display: flex;
    align-items: center;

    img {
      margin-right: 21px;
    }
  }
}
</style>
