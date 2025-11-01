<template>
  <el-header id="navbar">
    <div class="nav-left">
      <div class="logo" @click="goHome">
        <span>数据中心</span>
      </div>
    
    </div>
    <div class="nav-right">
      <div class="link" @click="goHome">
       返回首页
      </div>
    
    </div>
  </el-header>
</template>

<script lang="ts" setup>
import {ref, onMounted, watch} from 'vue';
import {useRouter} from 'vue-router'

const router = useRouter();

const goHome = () => {
  router.push('/')
}
</script>

<style lang="scss">
#navbar {
  .el-menu {
    min-width: 460px;
    background: transparent;
    border: none;
    height: 30px;

    .el-sub-menu__title, & > .el-menu-item {
      color: #d8d8d8;
    }
  }

  .el-menu--horizontal > .el-sub-menu .el-sub-menu__title:hover, .el-menu--horizontal .el-menu-item:not(.is-disabled):focus, .el-menu--horizontal .el-menu-item:not(.is-disabled):hover {
    background-color: transparent;
    border-radius: 2px;
  }

  .el-menu--horizontal > .el-menu-item.is-active {
    color: #FFFFFF !important;
    border-bottom-color: transparent;
    background-color: rgba(255, 255, 255, 0.2) !important;
    border-radius: 2px;
  }

  .el-menu--horizontal > .el-sub-menu.is-active .el-sub-menu__title {
    border-bottom-color: transparent;
    background-color: rgba(255, 255, 255, 0.2) !important;
    border-radius: 2px;
  }
}

.el-popover.el-popper.info-popper {
  background: #262E3E;
  box-shadow: 0px 3px 10px 0px rgba(0, 0, 0, 0.5);
  border-radius: 3px;
  border: none;
  padding: 20px 0 0;
  @include pageFont(14px, #FFFFFF, 400, "PingFangSC-Regular, PingFang SC", 20px);

  .el-popper__arrow {
    display: none;
  }

  .name, .number, ul {
    padding: 0 20px;
  }

  .number {
    margin: 10px 0 20px;
    display: flex;

    span:first-child {
      width: 30px;
    }

    span:last-child {
      max-width: 85%;
    }
  }

  ul {
    list-style-type: none;
    @include pageFont(12px, #FFFFFF, 400, "PingFangSC-Regular, PingFang SC", 20px);

    li {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 10px;

      & > div {
        width: 66%;
      }
    }

    .label {
      width: 60px;
    }

    .account-box {
      display: flex;
      align-items: center;
      justify-content: space-between;
    }

    .recharge {
      width: 64px;
      background: rgba(255, 255, 255, 0.2);
      border-radius: 5px;
      border: 1px solid rgba(4, 90, 166, 0.2);
      font-family: PingFangSC-Medium, PingFang SC;
      font-weight: 500;
      text-align: center;
      line-height: 27px;
      cursor: pointer;
    }
  }

  .signOut {
    text-align: center;
    padding: 10px 0;
    font-size: 12px;
    border-top: 1px solid rgba(255, 255, 255, 0.1);
    margin-top: 20px;
    cursor: pointer;
  }
}

.el-popper.nav-submenu-popper {
  box-shadow: 0px 3px 10px 0px rgba(0, 0, 0, 0.5);
  border-radius: 3px;
  border: none !important;
  background: #262E3E !important;

  .el-menu {
    background: #262E3E;
  }

  .el-menu--horizontal {
    border-bottom: none;
  }

  .el-menu--horizontal .el-menu .el-menu-item, .el-menu--horizontal .el-menu .el-sub-menu__title {
    background-color: transparent;
    @include pageFont(14px, #D8D8D8, 400, "PingFangSC-Regular, PingFang SC", 20px);
  }

  .el-menu--horizontal .el-menu .el-menu-item.is-active {
    background-color: rgba(255, 255, 255, 0.2);
    color: #FFFFFF;
  }
}
</style>
<style lang="scss" scoped>
#navbar {
  background: #262E3E;
  box-shadow: 0px 5px 10px 0px rgba(0, 0, 0, 0.3), 0px 2px 4px 0px rgba(0, 0, 0, 0.1);
  @include pageFont(14px, #D8D8D8, 400, "PingFangSC-Regular, PingFang SC", 20px);
  text-shadow: 0px 5px 10px rgba(0, 0, 0, 0.3);
  display: flex;
  justify-content: space-between;
}

.nav-left {
  display: flex;
  align-items: center;
}

.logo {
  width: 180px;
  cursor: pointer;
  font-size: 22px;
  display: flex;
  justify-content: center;
}

.nav-right {
  color: #CDCFD3;
  .link {
    min-width: 70px;
    cursor: pointer;
    margin-right: 40px;
    border-radius: 10px;
    border: 1px solid #666;
    text-align: center;
    padding: 5px 12px;
  }
}

.logo, .nav-right, .personal-item, .down-btn {
  display: flex;
  align-items: center;
}

.down-btn {
  width: 36px;
  min-height: 54px;
  background: url('/src/assets/images/nav_avbg.png') no-repeat;
  background-size: 100% 100%;
  background-position: center 5px;
  margin-right: 5px;
  justify-content: center;
  cursor: pointer;
}
</style>
