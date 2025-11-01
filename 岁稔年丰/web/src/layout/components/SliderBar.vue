<template>
  <!-- 菜单 -->
  <el-menu id="sliderbar" :default-active="activeIndex" :unique-opened="false" background-color="#1E222D"
           text-color="#D8D8D8" active-text-color="#FFFFFF">
    <ItemVue v-for="route in constantRoutes" :item="route" :base-path="route.path"
             :key="route.path ? route.path : route.meta?.title"></ItemVue>
  </el-menu>
</template>

<script lang="ts" setup>
import {ref, computed, watch} from 'vue';
import {useRouter} from "vue-router";
import constantRoutes from '../../router/console';
import ItemVue from './Item.vue';
//获取当前路由
let currentPath = "";
let router = useRouter()
watch(
    () => router.currentRoute.value.path, (newValue: any) => {
      currentPath = newValue;
    }, {immediate: true}
)
//获取默认activeIndex
let activeIndex = computed(() => {
  const route = useRouter();
  let meta = route.currentRoute.value.meta;
  let path = route.currentRoute.value.path;
  if (meta.activeMenu) {
    return meta.activeMenu;
  }
  return path;
})
</script>

<style lang="scss">
#sliderbar {
  background-color: transparent;
  border: none;

  .el-menu-item.is-active {
    background: #045AA6;
  }

  .el-sub-menu .el-menu-item {
    padding-left: 56px;
  }
}
</style>
